from typing import Union, Tuple

import json
import pathlib

import numpy as np

import torch

from .dataset import BaseDataset
from .dataset_utils import mean_downsample_cqt, normalize_cqt, upscale_cqt_values


class TestDataset(BaseDataset):
    """Test dataset.

    It flattens the versions of the cliques and returns their features as CQT features.
    The features are loaded from a directory containing the features as memmap files.
    The features can be downsampled in time by taking the mean, clipped to a dynamic range,
    padded if too short, and scaled to [0,1].
    """

    def __init__(
        self,
        cliques_json_path: str,
        features_dir: str,
        mean_downsample_factor: int = 20,
        cqt_bins: int = 84,
        scale: str = "norm",
        min_length: int = None,
    ) -> None:
        """Initializes the dataset

        Parameters:
        -----------
        cliques_json_path : str
            Path to the cliques json file
        features_dir : str
            Path to the directory containing the features
        mean_downsample_factor : int
            Factor by which to downsample the features by taking the mean
        cqt_bins : int
            Number of CQT bins in a feature array
        scale : str
            "normalize": scale the features to [0,1]
            "upscale": upscale the features (eg. used by CoverHunter)
        """

        assert cqt_bins > 0, f"Expected cqt_bins > 0, got {cqt_bins}"
        assert (
            mean_downsample_factor > 0
        ), f"Expected mean_downsample_factor > 0, got {mean_downsample_factor}"

        self.cliques_json_path = cliques_json_path
        self.features_dir = pathlib.Path(features_dir)
        self.mean_downsample_factor = mean_downsample_factor
        self.scale = scale
        self.cqt_bins = cqt_bins
        self.min_length = min_length

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

        # Determine the data source
        if "da-tacos" in cliques_json_path.lower():
            self.discogs_vi = False
            self.shs100k = False
            self.datacos = True
        elif "shs" in cliques_json_path.lower():
            self.discogs_vi = False
            self.shs100k = True
            self.datacos = False
        elif "discogs-vi" in cliques_json_path.lower():
            self.discogs_vi = True
            self.shs100k = False
            self.datacos = False
        else:
            raise ValueError("Dataset not recognized.")

        # In datacos all features are present so no need to filter
        if self.discogs_vi or self.shs100k:
            # Delete versions with missing features 
            print("Deleting versions with missing features...")
            self._delete_missing_features()

            # Count the number of cliques and versions again
            self.n_cliques, self.n_versions = 0, 0
            for versions in self.cliques.values():
                self.n_cliques += 1
                self.n_versions += len(versions)
            print(f"{self.n_cliques:>7,} cliques left.")
            print(f"{self.n_versions:>7,} versions left.")

        # Create a list of all versions together with their clique ID
        self.items = []
        if not self.datacos:
            for clique_id, versions in self.cliques.items():
                for i in range(len(versions)):
                    self.items.append((clique_id, i))
        else:
            for clique_id, versions in self.cliques.items():
                for version_id in versions.keys():
                    self.items.append((clique_id, version_id))

    def __getitem__(
        self, idx, encode_version=False
    ) -> Tuple[torch.Tensor, Union[str, int]]:
        """For a given index, returns the feature and label for the corresponding version.
        Features are loaded as full duration first and then downsampled but chunk sampling is
        not applied, i.e. a feature is returned in full duration.

        Parameters:
        -----------
        batch_idx: int
            Index of the version in the dataset
        encode_version: bool, optional
            If True, the label is a string in the format 'clique_id|version_id'.
            If False, the label is an integer obtained from the clique_id. Default is False.

        Returns:
        --------
        feature: torch.Tensor
            The CQT feature of the version shape=(F,T), dtype=float32

        label: Union[str, int]
            Label for the feature 'clique_id|version_id'. Depends on the value of encode_version.
            NOTE: Our labels are not consecutive integers. They are the clique_id.
        """
        clique_id, label, feature_id, feature_dir = self.get_feature_info(idx, encode_version)
        feature = self.load_cqt(feature_dir, feature_id, self.min_length)

        return feature, label

    def __len__(self) -> int:
        """Returns the number of versions in the dataset."""

        return len(self.items)
