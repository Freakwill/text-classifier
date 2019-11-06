#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections

import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg

from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

STOPWORDS = {'，', '。', '：', '；', '、', ',', '.', ' '}

def get_corpus(texts):
    return [' '.join(w.word for w in pseg.cut(text) if w.word not in STOPWORDS and  w.flag not in {'w','t','p','x', 'c', 'uj', 'd', 'un'}) for text in texts]

def get_data(corpus, vectorizer=TfidfVectorizer):
    # corpus = get_corpus(texts)
    vector = vectorizer(stop_words='english')
    vector.fit(corpus)
    vocabulary = list(vector.vocabulary_.keys())
    vocabulary.sort(key=lambda x:vector.vocabulary_[x])
    return vector.transform(corpus).todense()


def _lda(X, decomposition=LatentDirichletAllocation):
    # from scipy.special import softmax
    corpus = get_corpus(X)
    X = get_data(corpus)
    decomp = decomposition(n_components=5)
    decomp.fit(X.T)
    return np.array([[ci/sum(c)*len(x.split(' ')) for i, ci in enumerate(c)] for c, x in zip(decomp.components_.T, corpus)])


from sklearn.svm import SVC
from tpot import TPOTClassifier

class TextClassifier(BernoulliNB):

    def __init__(self, get_features=_lda, *args, **kwargs):
        super(TextClassifier, self).__init__(*args, **kwargs)
        self.get_features = get_features

    @classmethod
    def select(cls, model=GaussianNB):
        cls.__bases__ = model

    def fit_text(self, X, y):
        self.Xtrain = X
        self.ytrain = y
        self.n_samples = X.shape[0]

    def predict_text(self, X):
        Xtotal = np.hstack((self.Xtrain, X))
        Xtotal = self.get_features(Xtotal)
        self.fit(Xtotal[:self.n_samples], self.ytrain)
        return self.predict(Xtotal[self.n_samples:])

    def accuracy(self, X, y):
        X = self.get_features(X)
        return self.score(X, y)


if __name__ == '__main__':
    data = pd.read_csv('train.csv', sep='\t')
    X = data['comment']
    y = data['label']
    # TextClassifier.select()
    tc = TextClassifier()
    tc.fit_text(X, y)
    data = pd.read_csv('test.csv')
    y_ = tc.predict_text(data['comment'])
    df = pd.DataFrame(y_, columns=('label',), index=pd.Index(data['id'], name='id'))
    df.to_csv(f'result-{tc.accuracy(X, y):.4}.csv')
