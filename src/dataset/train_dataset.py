import json
import pathlib
from typing import Tuple
import random

import numpy as np
import torch

from .dataset import BaseDataset

class TrainDataset(BaseDataset):
    """Training dataset.

    It samples random anchor and positive versions from the same clique.
    The anchor and positive versions are returned as CQT features. The features are loaded from
    a directory containing the features as memmap files. The features can be downsampled in time
    by taking the mean, clipped to a dynamic range, padded if too short, and scaled to [0,1].
    An epoch is defined as when each clique is seen once, regardless of their number of versions.
    """

    def __init__(
        self,
        cliques_json_path: str,
        features_dir: str,
        max_length: int,
        min_length: int = None,
        mean_downsample_factor: int = 20,
        cqt_bins: int = 84,
        scale: str = "norm",
        clique_usage_ratio: float = 1.0,
        transform=None,
    ) -> None:
        """Initializes the training dataset.

        Parameters:
        -----------
        cliques_json_path : str
            Path to the cliques json file
        features_dir : str
            Path to the directory containing the features
        max_length : int
            Maximum length before downsampling. If a feature is longer than this, it is randomly cropped.
        min_length : int (optional)
            Minimum length of the context before downsampling. If a feature is shorter than this, its padded.
        mean_downsample_factor : int
            Factor by which to downsample the features by averaging
        cqt_bins : int
            Number of CQT bins in a feature array
        scale : str
            "normalize": scale the features to [0,1]
            "upscale": upscale the features (eg. used by CoverHunter)
        clique_usage_ratio: float
            Ratio of the cliques to use. If < 1.0, it will reduce the number of cliques.
            Usefull for debugging, short tests.
        transform : callable (optional)
            Augmentation pipeline.
        """

        assert max_length > 0, f"Expected max_length > 0, got {max_length}"
        assert (
            mean_downsample_factor > 0
        ), f"Expected mean_downsample_factor > 0, got {mean_downsample_factor}"
        assert cqt_bins > 0, f"Expected cqt_bins > 0, got {cqt_bins}"
        assert clique_usage_ratio > 0, f"Expected positive, got {clique_usage_ratio}"

        self.cliques_json_path = cliques_json_path
        self.features_dir = pathlib.Path(features_dir)
        self.max_length = max_length
        if min_length:
            self.min_length = min_length
        else:
            self.min_length = self.max_length
            
        self.mean_downsample_factor = mean_downsample_factor
        self.scale = scale
        self.cqt_bins = cqt_bins

        # Load the cliques
        print(f"Loading cliques from {cliques_json_path}")
        with open(cliques_json_path) as f:
            self.cliques = json.load(f)
            
        # Use a subset of cliques based on the clique_usage_ratio
        if clique_usage_ratio < 1.0:
            N = int(len(self.cliques) * clique_usage_ratio)
            sampled_ids = np.random.choice(list(self.cliques.keys()), N, replace=False)
            self.cliques = {clique_id: self.cliques[clique_id] for clique_id in sampled_ids}

        # Count the number of cliques and versions
        self.n_cliques, self.n_versions = 0, 0
        for versions in self.cliques.values():
            self.n_cliques += 1
            self.n_versions += len(versions)
        print(f"{self.n_cliques:>7,} cliques found.")
        print(f"{self.n_versions:>7,} versions found.")            

        # Delete versions with missing features
        print("Deleting versions with missing features...")
        self._delete_missing_features()

        self.clique_ids = list(self.cliques.keys())

        # Count the number of cliques and versions
        self.n_cliques, self.n_versions = 0, 0
        for versions in self.cliques.values():
            self.n_cliques += 1
            self.n_versions += len(versions)
        print(f"{self.n_cliques:>7,} cliques left.")
        print(f"{self.n_versions:>7,} versions left.")

        # TODO: clique usage alternative?

        self.clique_ids = [] # real labels, length is number of cliques
        self.clique_nums = [] # integer labels, length is number of cliques
        self.labels = [] # integer labels, length is the number of versions
        self.versions = [] # dict with metadata of version, length is the number of versions
        for i, (clique_id, versions) in enumerate(self.cliques.items()):
            self.clique_nums.append(i)
            for version in versions:
                self.clique_ids.append(clique_id)
                self.labels.append(i)
                self.versions.append(version)
        
        self.transform = transform
     
    def __getitem__(self, index) -> Tuple[torch.Tensor, list]:
        """Get self.samples_per_clique random anchor versions from a given clique.

        Parameters:
        -----------
        index : int
            Index of the clique to sample versions from
        Returns:
        --------
        anchors : torch.Tensor
            CQT features of self.samples_per_clique versions.
            shape=(self.samples_per_clique, F, T), dtype=float32
            see self.load_cqt for more details.
        labels : list
            List of labels. The content depends on the encode_version parameter.
        """
        label = self.labels[index]
        # Get feature
        version = self.versions[index]
        
        feature_dir = self.features_dir / version["youtube_id"][:2]
        feature = self.load_cqt(feature_dir=feature_dir, feature_id=version["youtube_id"], 
                                min_length=self.min_length, max_length=self.max_length)
        
        if self.transform:
            feature = self.transform(feature)

        return feature, label

    def __len__(self) -> int:
        """Each version appears once at each epoch."""

        return len(self.labels)

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