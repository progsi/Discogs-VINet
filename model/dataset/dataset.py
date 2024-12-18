import pathlib
import json
from collections import defaultdict

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """BaseDataset with elementary functions.
    """

    def __init__(
        self,
        cliques_json_path: str,
        features_dir: str,
        mean_downsample_factor: int = 20,
        cqt_bins: int = 84,
        scale: bool = True,
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
        scale : bool
            Whether to scale the features to [0,1]
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
