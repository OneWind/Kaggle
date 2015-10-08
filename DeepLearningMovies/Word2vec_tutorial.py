# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:11:33 2015

@author: fyi
"""

# import modules & set up logging
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)


import os
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('/Users/fyi/Documents/kaggle/DeepLearningMovies/Data') # a memory-friendly iterator

model = gensim.models.Word2Vec(sentences, 
                               min_count=100,
                               size=100,
                               workers=2)

model.accuracy("/Users/fyi/Documents/kaggle/DeepLearningMovies/questions-words.txt")

model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
model.doesnt_match("breakfast cereal dinner lunch".split())
model.similarity('woman', 'man')
