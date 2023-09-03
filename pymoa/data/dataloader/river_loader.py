from river.datasets.base import Dataset
from .base import DataLoader

class RiverLoader(DataLoader):
    def __init__(self, dataset: Dataset,
                 total_instances: int = 1000,
                 decay_horizon: int = 1000):
        super().__init__(dataset, decay_horizon)
        self.length = int(total_instances)

    def __iter__(self):
        for x, y in self.dataset.take(self.length):
            x_values = list(x.values())
            self._decay_dataset.append((x_values, y))
            yield x_values, y
            
    def __len__(self):
        return self.length
