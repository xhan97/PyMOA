from collections import deque

class DataLoader():
    def __init__(self, dataset, decay_horizon: int = 1000):
        self.dataset = dataset
        self._decay_dataset = deque(maxlen=decay_horizon)
        self.m_timestamp = 0

    def __iter__(self):
        for x, y in self.dataset:
            self._decay_dataset.append((x, y))
            self.m_timestamp += 1
            yield x, y

    def __len__(self):
        return self.m_timestamp

    def __getitem__(self, idx):
        return self._decay_dataset[idx]
    
    @property
    def decay_dataset(self):
        X = [x for x, y in self._decay_dataset]
        Y = [y for x, y in self._decay_dataset]
        return {'X': X, 'y': Y}
    
    
