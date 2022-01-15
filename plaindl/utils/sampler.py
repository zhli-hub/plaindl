import numpy as np
import collections


class UnigramSampler(object):
    def __init__(self, corpus, power, sample_size):
        self.word_p = None
        self.sample_size = sample_size
        self.vocab_size = None

        counts = collections.Counter(corpus).most_common()

        self.vocab_size = len(counts)
        self.word_p = np.zeros(self.vocab_size)

        for i in range(self.vocab_size):
            self.word_p[counts[i][0]] = counts[i][1]

        self.word_p = np.power(self.word_p, power)
        self.word_p = self.word_p / np.sum(self.word_p)


    def get_negative_sample(self, target):
        assert target.ndim == 1
        batch_size = target.size

        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):
            p = self.word_p.copy()
            p[target[i]] = 0
            p = p / np.sum(p)
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        return negative_sample




if __name__ == '__main__':
    corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3, 1, 1, 1, 1])
    sampler = UnigramSampler(corpus, 0.75, 2)
    target = np.array([4, 3, 0])
    print(target.size)
    print(sampler.get_negative_sample(target))
