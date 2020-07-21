from random import choice

from src.base.sampler import Sampler


class RandomSampler(Sampler):
    def sample(self):
        return self.random_sample()

    def get_init(self):
        return 0

    def to_string(self):
        return "random"
