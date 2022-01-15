import numpy as np
import os
import gzip
import plaindl.utils.functional as F
import plaindl.nn.nlp.functional as NLP_F
import pickle
try:
    import urllib.request
except ImportError:
    raise ImportError('Use Python3!')


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class PTB(Dataset):
    url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
    key_file = {
        'train': 'ptb.train.txt',
        'test': 'ptb.test.txt',
        'valid': 'ptb.valid.txt'
    }
    save_file = {
        'train': 'ptb.train.npy',
        'test': 'ptb.test.npy',
        'valid': 'ptb.valid.npy'
    }
    vocab_file = 'ptb.vocab.pkl'

    def _download(self, file_name):
        file_path = self.root + '/' + file_name
        if os.path.exists(file_path):
            return

        print('Downloading ' + file_name + ' ... ')

        try:
            urllib.request.urlretrieve(PTB.url_base + file_name, file_path)
        except urllib.error.URLError:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(PTB.url_base + file_name, file_path)

        print('Done')

    def load_vocab(self):
        vocab_path = self.root + '/' + PTB.vocab_file
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                self.word_to_id, self.id_to_word = pickle.load(f)

        self.words = open(self.file_path).read().replace('\n', '<eos>').strip().split()

        for i, word in enumerate(self.words):
            if word not in self.word_to_id:
                tmp_id = len(self.word_to_id)
                self.word_to_id[word] = tmp_id
                self.id_to_word[tmp_id] = word

        with open(vocab_path, 'wb') as f:
            pickle.dump((self.word_to_id, self.id_to_word), f)



    def __init__(self, root, data_type, window_size=1, download=False):
        """

        :param root:读取数据的路径，若该路径内没有数据，则会将数据下载到该路径内
        :param data_type:读取数据的类型：'train' or 'test' or 'valid (val)'
        """
        self.root = root
        self.download = download
        self.data_type = data_type
        self.word_to_id = {}
        self.id_to_word = {}
        self.words = {}

        if self.data_type == 'val':
            self.data_type = 'valid'
        self.save_path = self.root + '/' + PTB.save_file[self.data_type]
        self.file_path = self.root + '/' + PTB.key_file[self.data_type]

        if download:
            self._download(self.file_path)
        else:
            if not os.path.exists(self.file_path):
                raise RuntimeError('PTB data not found.You can use download=True to download it.')

        self.load_vocab()

        if os.path.exists(self.save_path):
            self.corpus = np.load(self.save_path)
        else:
            self.corpus = np.array([self.word_to_id[w] for w in self.words])
            np.save(self.save_path, self.corpus)

        self.contexts, self.target = NLP_F.create_contexts_target(self.corpus, window_size)

    def __getitem__(self, index):
        return self.contexts[index], self.target[index]

    def __len__(self):
        return len(self.contexts)

    def get_data(self):
        return self.corpus, self.word_to_id, self.id_to_word


class MNIST(Dataset):
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
    ]

    def _download(self):
        pass

    def _check_exists(self):
        return True

    def _load(self, path):

        with gzip.open(path[1], 'rb') as lbpath:
            y = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        if self.one_hot:
            y = F.to_one_hot(y)

        with gzip.open(path[0], 'rb') as imgpath:
            x = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y), 28*28)
        return x, y

    def __init__(self, root, train=True, transform=None, one_hot=True, download=False):
        self.root = root
        self.train = train
        self.one_hot = one_hot
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError('MNIST data not found.You can use download=True to download it.')

        self.paths = []

        for fname in MNIST.files:
            self.paths.append((os.path.join(self.root, fname)))

        if train:
            self.x_train, self.y_train = self._load([self.paths[0], self.paths[1]])

        else:
            self.x_test, self.y_test = self._load([self.paths[2], self.paths[3]])

        # print(self.x_test.shape)

    def __getitem__(self, index):
        if self.train:
            return self.x_train[index], self.y_train[index]
        return self.x_test[index], self.y_test[index]

    def __len__(self):
        if self.train:
            return len(self.y_train)
        return len(self.y_test)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    mnist_data = MNIST('../../example/mnist_data', train=True, one_hot=True)

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(30):
        images, label = mnist_data[i]

        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(images.reshape(28, 28), cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(label))
    plt.show()
