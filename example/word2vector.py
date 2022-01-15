import plaindl as pdl
from plaindl.nn.nlp.cbow import CBOW
from plaindl.data.datasets import PTB
from plaindl.data.dataloader import DataLoader
import pickle

import matplotlib.pyplot as plt
import numpy as np

window_size = 1
hidden_size = 100
batch_size = 32
max_epoch = 10

train_dataset = PTB('./PTB', data_type='train', window_size=window_size, download=False)
corpus, word_to_id, id_to_word = train_dataset.get_data()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
model = CBOW(train_dataloader.size, hidden_size=hidden_size, window_size=window_size, corpus=corpus)

max_iters = train_dataloader.size // batch_size

optim = pdl.optim.SGD(1)
loss_list = []

for epoch in range(max_epoch):
    iters = 0
    total_loss = 0
    loss_count = 0

    for data in train_dataloader:
        contexts, target = data

        loss = model(contexts, target)
        optim.gradient(loss)
        optim.update()

        total_loss += loss.value
        loss_count += 1

        if (iters + 1) % 2 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d |  iter %d / %d | loss %.6f'
                  % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0
        iters += 1

plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iterations (x2)')
plt.ylabel('loss')
plt.show()

word_vecs = model.word_vecs

params = {'word_vecs': word_vecs.astype(np.float16), 'word_to_id': word_to_id, 'id_to_word': id_to_word}
pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
