import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier

import csv

from sklearn.cross_validation import KFold, StratifiedKFold

import numpy as np

#clasifiers
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing.label import label_binarize

from stop_words import get_stop_words

from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt
from main.utilities import Util

class KNN:
    
    accuracy = []
    
    def __init__(self):
        self.accuracy = []
        
    def get_accuracy(self):
        return self.accuracy

    def getPredicted(self):
        return self.predicted
        
    def go(self):
        stopWords = get_stop_words('english')
        stopWords += ['said','will','three','say','also','one','good','well','made','take','uk','fawn']
        
        vectorizer=CountVectorizer(stop_words = stopWords,strip_accents='unicode',max_df=0.95,analyzer='word')
#         vectorizer=TfidfVectorizer(stop_words = stopWords,strip_accents='unicode',sublinear_tf=True,use_idf=True,analyzer='word')
        transformer=TfidfTransformer()
        clf = KNeighborsClassifier(n_neighbors=5)

        df=pd.read_csv("train_set.csv",sep="\t")
        df2=pd.read_csv('probasForDataset.csv',sep="\t")
        
        le = preprocessing.LabelEncoder()
        le.fit(df2["topicNumber"])
        Y=le.transform(df2["topicNumber"])
        title = df['Title']
        X = df['Content']
        X += 10*title.copy()
        tempX = X
        X = vectorizer.fit_transform(tempX)
        X_transformed =  transformer.fit_transform(X)
        svd=TruncatedSVD(n_components=100, random_state=42)
        X_lsi=svd.fit_transform(X_transformed)
        X = X_lsi
        
        self.accuracy = []
        
        u = Util()
        
        cv = StratifiedKFold(Y, n_folds=10)
        for i, (train, test) in enumerate(cv):
            
            print 'Calculating KNeighborsClassifier...'
            clf.fit(X[train],Y[train])
            predicted = clf.predict(X[test])
            accuracyKNeighborsClassifier = accuracy_score(Y[test],predicted)
            self.accuracy.append(accuracyKNeighborsClassifier)
            print "Finished calculation KNeighborsClassifier!"
            
            u.calcRoc(le, Y, test,predicted)
            
    
    def goWithInput(self,X,Y,train,test):
        stopWords = get_stop_words('english')
        stopWords += ['said','will','three','say','also','one','good','well','made','take','uk','fawn']
        
        # vectorizer=CountVectorizer(stopWords)
        vectorizer=TfidfVectorizer(stop_words = stopWords,strip_accents='unicode',sublinear_tf=True,use_idf=True,analyzer='word')
        transformer=TfidfTransformer()
        clf = KNeighborsClassifier(n_neighbors=5)
        
        tempX = X
        X = vectorizer.fit_transform(tempX)
        X_transformed =  transformer.fit_transform(X)
        X = X_transformed
        svd=TruncatedSVD(n_components=10, random_state=42)
        X_lsi=svd.fit_transform(X_transformed)
        X = X_lsi
        
        X_train = X[train]
        Y_train = Y[train]
        X_test = X[test]
        Y_test = Y[test]
        
        print 'Calculating KNeighborsClassifier...'
        clf.fit(X_train,Y_train)
        predicted = clf.predict(X_test)
        accuracyKNeighborsClassifier = accuracy_score(Y_test,predicted)
        self.accuracy.append(accuracyKNeighborsClassifier)
        self.predicted = predicted
        print "Finished calculation KNeighborsClassifier!"
        
    def goForTestCsv(self):
        stopWords = get_stop_words('english')
        stopWords += ['said','will','three','say','also','one','good','well','made','take','uk','fawn']
        
        vectorizer=CountVectorizer(stop_words = stopWords,strip_accents='unicode',max_df=0.95,analyzer='word')
#         vectorizer=TfidfVectorizer(stop_words = stopWords,strip_accents='unicode',sublinear_tf=True,use_idf=True,analyzer='word')
        transformer=TfidfTransformer()
        clf = KNeighborsClassifier(n_neighbors=5)

        df=pd.read_csv("train_set.csv",sep="\t")
        df_test = pd.read_csv("test_set.csv",sep="\t")
        df_lda = pd.read_csv('X_train_10.csv',sep="\t")
        
        le = preprocessing.LabelEncoder()
        le.fit(df["Category"])
        Y=le.transform(df["Category"])
        title = df['Title']
        X = df_lda['Content']
        X += 10*title.copy()
        tempX = X
        X = vectorizer.fit_transform(tempX)
        X_transformed =  transformer.fit_transform(X)
        svd=TruncatedSVD(n_components=100, random_state=42)
        X_lsi=svd.fit_transform(X_transformed)
        X = X_lsi
        
        title_test = df_test['Title']
        X_test = df_test['Content']
        X_test += 10*title_test.copy()
        tempX = X_test
        X_test = vectorizer.transform(tempX)
        X_transformed =  transformer.transform(X_test)
        X_test_lsi=svd.transform(X_transformed)
        X_test = X_test_lsi
        
        self.accuracy = []
            
        print 'Calculating KNeighborsClassifier...'
        clf.fit(X,Y)
        self.predicted = clf.predict(X_test)
        print "Finished calculation KNeighborsClassifier!"
        
        return le.inverse_transform(self.predicted)





















