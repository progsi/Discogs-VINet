from collections import defaultdict
import random 
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_utils import mean_downsample_cqt, normalize_cqt, upscale_cqt_values

GENRES_KEY = "release_genres"
STYLES_KEY = "release_styles"
COUNTRY_KEY = "country"
YEAR_KEY = "released"  

class BaseDataset(Dataset):
    """BaseDataset with elementary functions.
    """
    def _delete_missing_features(self):
        """Deletes versions with missing features and afterwards,
        cliques with less than 2 versions.
        """
        yt_ids = [version["youtube_id"] for clique in self.cliques.values() for version in clique]
        clique_ids = [clique_id for clique_id, versions in self.cliques.items() for _ in versions]
        # Initialize the keep list with the same length as yt_ids, defaulting to True
        keep = [(self.features_dir / yt_id[:2] / f"{yt_id}.mm").exists() for yt_id in yt_ids]

        # Check if each clique_id has at least two entries with True in keep
        clique_count = defaultdict(int)
        for i, clique_id in enumerate(clique_ids):
            if keep[i]:
                clique_count[clique_id] += 1

        # Set the whole set of indices of that clique_id to False if it has less than 2 True entries
        for i, clique_id in enumerate(clique_ids):
            if clique_count[clique_id] < 2:
                keep[i] = False
        
        # Delete items in self.cliques based on the keep list
        index = 0
        for clique_id, versions in list(self.cliques.items()):
            self.cliques[clique_id] = [version for i, version in enumerate(versions) if keep[index + i]]
            index += len(versions)
            # Remove cliques without remaining values
            if not self.cliques[clique_id]:
                del self.cliques[clique_id]
                
    def get_int_label(self, clique_id: str) -> int:
        return int(clique_id.split("W_")[1])
    
    def get_feature_info(self, idx, encode_version=True):
        if self.datacos:
            clique_id, version_id = self.items[idx]
            if encode_version:
                label = f"{clique_id}|{version_id}"
            else:
                label = self.get_int_label(idx)
            feature_id = version_id
            feature_dir = self.features_dir
        else:
            clique_id, version_idx = self.items[idx]
            version_dict = self.cliques[clique_id][version_idx]
            if encode_version:
                label = f'{clique_id}|{version_dict["version_id"]}'
            else:
                if self.discogs_vi:
                    label = int(clique_id.split("C-")[1])
                else:
                    label = int(clique_id)
            feature_id = version_dict["youtube_id"]
            feature_dir = self.features_dir / feature_id[:2]
        return clique_id, label, feature_id, feature_dir
            
    def load_cqt(self, 
                 feature_dir: str,
                 feature_id: str, 
                 min_length: int = None, 
                 max_length: int = None) -> torch.Tensor:
        """Load the magnitude CQT features for a single version with yt_id from the features
        directory. Loads the memmap file, if the feature is long enough it takes a random
        chunk or pads the feature if it is too short. Then it downsamples the feature in time,
        clips the feature to the dynamic range, and scales the feature if specified
        Converts to torch tensor float32.

        Parameters:
        -----------
        yt_id : str
            The YouTube ID of the version

        Returns:
        --------
        feature : torch.FloatTensor
            The CQT feature of the version dtype=float32, shape: (F, T)
            T is the downsampled context length, which is determined during initialization.
        """
        # We store the features as a memmap file
        feature_path = feature_dir / f"{feature_id}.mm"
        # And the shape as a separate numpy array
        feature_shape_path = feature_dir / f"{feature_id}.npy"
        # Load the memmap shape
        feature_shape = tuple(np.load(feature_shape_path, mmap_mode="r"))

        # Load the magnitude CQT
        if min_length and max_length:
            length = random.randint(min_length, max_length)
        else:
            length = feature_shape[0]
    
        if feature_shape[0] > length:
            # If the feature is long enough, take a random chunk
            start = np.random.randint(0, feature_shape[0] - length)
            fp = np.memmap(
                feature_path,
                dtype="float16",
                mode="r",
                shape=(length, feature_shape[1]),
                offset=start * feature_shape[1] * 2,  # 2 bytes per float16
            )
        else:
            # Load the whole feature
            fp = np.memmap(
                feature_path,
                dtype="float16",
                mode="r",
                shape=feature_shape,
            )

        # Convert to float32
        feature = np.array(fp, dtype=np.float32)  # (T, F)
        del fp
        assert feature.size > 0, "Empty feature"
        assert feature.ndim == 2, f"Expected 2D feature, got {feature.ndim}D"
        assert (
            feature.shape[1] == self.cqt_bins
        ), f"Expected {self.cqt_bins} features, got {feature.shape[1]}"

        # Pad the feature if it is too short
        if min_length and feature.shape[0] < min_length:
            feature = np.pad(
                feature,
                ((0, min_length - feature_shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Downsample the feature in time
        if self.mean_downsample_factor > 1:
            feature = mean_downsample_cqt(feature, self.mean_downsample_factor)

        # Clip the feature below zero to be sure
        feature = np.where(feature < 0, 0, feature)

        # Scale the feature to [0,1] if specified
        if self.scale == "normalize":
            feature = normalize_cqt(feature)
        elif self.scale == "upscale":
            feature = upscale_cqt_values(feature)

        # Transpose to (F, T) because the CQT is stored as (T, F)
        feature = feature.T

        # Convert to tensor (view not a copy)
        feature = torch.from_numpy(feature)

        return feature

    @staticmethod
    def collate_fn(items) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function for the dataset. 
        Parameters:
        -----------
        items: list
            List of tuples containing the features and labels of the versions

        Returns:
        --------
        features: torch.Tensor
            The CQT features of the anchors, shape=(B, F, T), dtype=float32
            F is the number of CQT bins.
            T is the downsampled max_length,
        labels: torch.Tensor
            1D tensor of the clique labels, shape=(B,)
        """
        
        # padding
        max_length = max(item[0].shape[1] for item in items)
        padded_features = [
            torch.nn.functional.pad(item[0], (0, max_length - item[0].shape[1])) for item in items
        ]
        
        features = torch.stack(padded_features)
        labels = torch.tensor([item[1] for item in items])

        return features, labels
