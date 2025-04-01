import numpy as np

eps = 1e-8

class StandardScaler:
    def __init__(self):
        pass

    def fit(self, x):
        self.mu = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.len = len(self.mu)
        self.std = np.maximum(self.std, eps)

    def transform(self, x):
        return (x - self.mu) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def un_transform(self, x):
        return x * self.std + self.mu

    def un_transform_var(self, x):
        return x * self.std