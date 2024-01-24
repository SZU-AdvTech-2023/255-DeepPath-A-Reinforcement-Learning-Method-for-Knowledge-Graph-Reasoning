import numpy as np

# 新增的状态正则化工具，好像用处不大
class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if x is None:
            return x
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        # print("self.running_ms.mean",self.running_ms.mean)

        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x