from collections import defaultdict

from torch.utils.data import Dataset

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
