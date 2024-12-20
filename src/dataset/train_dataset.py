import json
import pathlib
from typing import Tuple

import numpy as np
import torch

from .dataset import BaseDataset
from .dataset_utils import mean_downsample_cqt

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
        context_length: int,
        mean_downsample_factor: int = 20,
        cqt_bins: int = 84,
        scale: bool = True,
        clique_usage_ratio: float = 1.0,
    ) -> None:
        """Initializes the training dataset.

        Parameters:
        -----------
        cliques_json_path : str
            Path to the cliques json file
        features_dir : str
            Path to the directory containing the features
        context_length : int
            Length of the context before downsampling. If a feature is longer than this,
            a chunk of this length is taken randomly. If a feature is shorter, it is padded.
        mean_downsample_factor : int
            Factor by which to downsample the features by averaging
        cqt_bins : int
            Number of CQT bins in a feature array
        scale : bool
            Whether to scale the features to [0,1]
        clique_usage_ratio: float
            Ratio of the cliques to use. If < 1.0, it will reduce the number of cliques.
            Usefull for debugging, short tests.
        """

        assert context_length > 0, f"Expected context_length > 0, got {context_length}"
        assert (
            mean_downsample_factor > 0
        ), f"Expected mean_downsample_factor > 0, got {mean_downsample_factor}"
        assert cqt_bins > 0, f"Expected cqt_bins > 0, got {cqt_bins}"
        assert clique_usage_ratio > 0, f"Expected positive, got {clique_usage_ratio}"

        self.cliques_json_path = cliques_json_path
        self.features_dir = pathlib.Path(features_dir)
        self.context_length = context_length
        self.mean_downsample_factor = mean_downsample_factor
        self.scale = scale
        self.cqt_bins = cqt_bins

        # Load the cliques
        print(f"Loading cliques from {cliques_json_path}")
        with open(cliques_json_path) as f:
            self.cliques = json.load(f)

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

        self.clique_ids = [] # real labels
        self.labels = [] # integer labels
        self.versions = [] # dict with metadata of version
        for i, (clique_id, versions) in enumerate(self.cliques.items()):
            for version in versions:
                self.clique_ids.append(clique_id)
                self.labels.append(i)
                self.versions.append(version)
     
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
        feature = self.load_cqt(version["youtube_id"])

        return feature, label

    def __len__(self) -> int:
        """Each version appears once at each epoch."""

        return len(self.labels)

    def load_cqt(self, yt_id) -> torch.Tensor:
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

        # Get the directory of the features
        feature_dir = self.features_dir / yt_id[:2]
        # We store the features as a memmap file
        feature_path = feature_dir / f"{yt_id}.mm"
        # And the shape as a separate numpy array
        feature_shape_path = feature_dir / f"{yt_id}.npy"
        # Load the memmap shape
        feature_shape = tuple(np.load(feature_shape_path, mmap_mode="r"))

        # Load the magnitude CQT
        if feature_shape[0] > self.context_length:
            # If the feature is long enough, take a random chunk
            start = np.random.randint(0, feature_shape[0] - self.context_length)
            fp = np.memmap(
                feature_path,
                dtype="float16",
                mode="r",
                shape=(self.context_length, feature_shape[1]),
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
        if feature.shape[0] < self.context_length:
            feature = np.pad(
                feature,
                ((0, self.context_length - feature_shape[0]), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Downsample the feature in time
        if self.mean_downsample_factor > 1:
            feature = mean_downsample_cqt(feature, self.mean_downsample_factor)

        # Clip the feature below zero to be sure
        feature = np.where(feature < 0, 0, feature)

        # Scale the feature to [0,1] if specified
        if self.scale:
            feature /= (
                np.max(feature) + 1e-6
            )  # Add a small value to avoid division by zero

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
            T is the downsampled context_length,
        labels: torch.Tensor
            1D tensor of the clique labels, shape=(B,)
        """

        features = torch.stack([item[0] for item in items])
        labels = torch.tensor([item[1] for item in items])

        return features, labels