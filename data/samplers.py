import random

import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class NormDiscrete:

    def __init__(self, config, infimum, supremum,seed=2021):
        random.seed(seed)
        np.random.seed(seed)
        self.config = config
        self.infimum = 1 if infimum < 1 else infimum
        self.supremum = supremum
        self.loc = (self.supremum + self.infimum) / 2

    def discrete_norm(self, size=1, plot=False):
        x = np.arange(self.infimum, self.supremum)
        xU, xL = x + 0.5, x - 0.5
        # create the probability regions using the normal distribution in the given interval with mean the center of the interval and variance such as P(infimum<x<supremum)~= 1
        prob = ss.norm.cdf(xU, loc=self.loc, scale=(self.supremum + 1 - self.loc) / 3) - ss.norm.cdf(xL, loc=self.loc, scale=(self.supremum + 1 - self.loc) / 3)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        nums = np.random.choice(x, size=size, p=prob)

        if plot:
            fig, ax = plt.subplots()
            sns.histplot(nums, discrete=True, ax=ax)
            ax.set_xlim(0, 11)
            ax.set_xticks(range(1, 11))
            plt.show()
        # if 0 make it one
        for num in nums:
            if num == 0:
                num = 1

        return nums
