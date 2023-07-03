"""
Further information:
https://github.com/llSourcell/logistic_regression/blob/master/Sentiment%20analysis%20with%20Logistic%20Regression.ipynb
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer

from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import  naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords


nltk.download('stopwords')

stop = stopwords.words('english')


ppath = os.path.join(os.path.abspath(''))

data = pd.read_csv(os.path.join(ppath, 'data/github_gold.csv'), delimiter=';')

porter = PorterStemmer()

def tokenizer(txt):
    return [porter.stem(word) for word in txt.split()] 


#Create a bag of words dictionary from the data.

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=True,
                        preprocessor=None,
                        stop_words=stop)


#Create train and test splits.
#Stratify makes sure that it's the same distrubition in the train and test split, as in the original data set.

X = data.Text
Y = data.Polarity
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0, stratify=Y)


param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, Y_train)

print()
print()
print('~~~ Logistic Regression ~~~')
print('Best parameter set: ' + str(gs_lr_tfidf.best_params_))
print('Best accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Accuracy in test: %.3f' % clf.score(X_test, Y_test))

print('Grid Scores:')
pprint(gs_lr_tfidf.cv_results_['mean_test_score'])
print()

X = data.Text
Y = data.Polarity
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0, stratify=Y)


param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, None],
               'clf__alpha': [1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', naive_bayes.MultinomialNB())])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, Y_train)


print()
print('~~~ Naive Bayes ~~~')
print('Best parameter set: ' + str(gs_lr_tfidf.best_params_))
print('Best accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Accuracy in test: %.3f' % clf.score(X_test, Y_test))

print('Grid Scores:')
pprint(gs_lr_tfidf.cv_results_['mean_test_score'])
print()


X = data.Text
Y = data.Polarity
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0, stratify=Y)


param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, None],
               'clf__degree': [1, 2, 3, 4, 5],
               'clf__C': [1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', svm.SVC())])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, Y_train)

print()
print('~~~ SVM ~~~')
print('Best parameter set: ' + str(gs_lr_tfidf.best_params_))
print('Best accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Accuracy in test: %.3f' % clf.score(X_test, Y_test))

print('Grid Scores:')
pprint(gs_lr_tfidf.cv_results_['mean_test_score'])
print()
