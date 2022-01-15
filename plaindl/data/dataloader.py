import numpy as np


class _DataLoaderIter(object):
    def __init__(self,dataloader):
        self.dataloader = dataloader
        self.max_iters = len(dataloader.dataset) // dataloader.batch_size
        self.batch_size = dataloader.batch_size
        if dataloader.shuffle:
            self.load_order = np.random.permutation(len(dataloader.dataset))
        else:
            self.load_order = range(len(dataloader.dataset))
        # print(self.load_order)

        self.current = 0

    def __next__(self):
        if self.current < self.max_iters:

            inputs, labels = (self.dataloader.dataset[self.load_order[self.current:self.batch_size + self.current]])
            self.current = self.current + 1

            return inputs, labels

        else:
            raise StopIteration



    def __iter__(self):
        return self


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    @property
    def size(self):
        return len(self.dataset)

    def __iter__(self):
        return _DataLoaderIter(self)


if __name__ == '__main__':
    import datasets
    trainDataLoader = DataLoader(dataset=datasets.MNIST('../../example/mnist_data', train=True, one_hot=True),
                                 batch_size=64,
                                 shuffle=False)
    print(trainDataLoader.size)

    for data in trainDataLoader:
        x, y = data


