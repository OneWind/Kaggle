# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:37:16 2015

@author: fyi
"""

import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

review = []

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   


test = pd.read_csv("/Users/fyi/Documents/kaggle/DeepLearningMovies/Data/testData.tsv", 
                   header=0, delimiter="\t", quoting=3)
for index, row in test.iterrows():
    review.append(row['review'])

train = pd.read_csv("/Users/fyi/Documents/kaggle/DeepLearningMovies/Data/labeledTrainData.tsv", 
                   header=0, delimiter="\t", quoting=3)
for index, row in train.iterrows():
    review.append(row['review'])

trainnolab = pd.read_csv("/Users/fyi/Documents/kaggle/DeepLearningMovies/Data/unlabeledTrainData.tsv", 
                   header=0, delimiter="\t", quoting=3)
for index, row in trainnolab.iterrows():
    review.append(row['review'])

review_cleaned = [review_to_words(w) for w in review]

with open('/Users/fyi/Documents/kaggle/DeepLearningMovies/Data/allreview.txt', 'w') as f:
    for rv in review:
        f.write("%s\n" % rv)

f.close()
