import nltk
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
import json

nltk.download('stopwords') 
nltk.download('punkt') 
stoplist = stopwords.words('english')


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(counter,threshold=3):
    # Ignore rare words
    words = [[cnt,word] for word, cnt in counter.items() if ((cnt >= threshold))]
    words.sort(reverse=True)
    words = [e[1] for e in words[:50000]]
    # f = open('vocab_file.txt','w')

    # Create a vocabulary and initialize with special tokens
    vocab = Vocabulary()
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')

    # Add the all the words
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

if __name__ == '__main__':
    emb = dict()
    f = open('glove.6B.200d.txt','r',encoding='utf-8')
    e = f.readline()
    while e:
        line = e.split(' ')
        emb[line[0]] = line[1:]
        e = f.readline()
    
    counter = Counter()
    with open('data/train.json', 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            doc = " ".join(data["document"])
            tokens = nltk.tokenize.word_tokenize(doc.lower())
            counter.update(tokens)


    vocab = build_vocab(counter)
  
    np.save('vocab_inspec.npy', vocab)
    
   
