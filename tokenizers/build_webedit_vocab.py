from cgi import test
import sys

from numpy import sort
# sys.path.append('../')

from data import WebEditingLSTMDataset

train_set = WebEditingLSTMDataset(None, None, 'train')
test_set = WebEditingLSTMDataset(None, None, 'test')

# all_vocab = train_set.vocab | test_set.vocab
from collections import Counter
import pdb

# print(Counter(train_set.vocab))
# print(Counter(test_set.vocab))

all_entitys = set(train_set.entitys)
all_vocab = set(train_set.vocab) - all_entitys
pdb.set_trace()

counter = Counter(train_set.entitys)
for vocab in counter:
    freq = counter[vocab]
    if freq > 10:
        all_vocab.add(vocab)


# pdb.set_trace()

all_vocab = all_vocab - {'<H>', '<R>', '<T>', '<bos>', '<eos>'}

all_vocab = sorted(all_vocab)

words = []
for word in ['<pad>', '<unk>', '<bos>', '<eos>', '<H>', '<R>', '<T>']:
    words.append(word)
for word in all_vocab:
    words.append(word)

    if '\n' in words:
        print('123')

print(len(words))

with open('webedit_vocab2.txt', 'w') as f:
    f.write('\n'.join(words))

