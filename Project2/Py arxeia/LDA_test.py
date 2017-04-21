import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import metrics

# from main.naiveBayes import NB
# from main.knn import KNN
# from main.svc import SVC
# from main.randomForest import RF

import csv

from sklearn.cross_validation import StratifiedKFold

#clasifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.classification import accuracy_score


with open("./tpt.csv", 'wb') as csvfile:
    writer = csv.writer(csvfile,delimiter=' ',quoting=csv.QUOTE_MINIMAL)
    writer.writerow('RowNum\tId\tTitle\tContent\tCategory')
    
    #Read Data
    df=pd.read_csv("./train_set.csv",sep="\t")
    df2 = pd.read_csv('testLdaWords_10.csv',sep='\t')
    le = preprocessing.LabelEncoder()
    le.fit(df["Category"])
    Y_train=le.transform(df["Category"])
    X_train=df['Content']
    
    X = df2['words']
    
#     X_train += 20*X.copy()
    
    i=0;
    for x in X:
        if not x:
            X_train[i] = ''
            i+=1
            continue
        X_train[i] += 20*X[i]
        i+=1
           
        if( i%200 == 0):
            print i
    
    
    raw_data = {'Content':X_train}
    df2 = pd.DataFrame.from_dict(raw_data)   
    df2.to_csv('X_train_test_10.csv',sep="\t", encoding='utf-8')
    vectorizer=CountVectorizer(stop_words='english')
    transformer=TfidfTransformer()
    svd=TruncatedSVD(n_components=10, random_state=42)
#     clf=SGDClassifier()
    clf = RandomForestClassifier()
#     lda=LatentDirichletAllocation(n_topics=10, max_iter=5,
#                                 learning_method='online', learning_offset=50.,
#                                 random_state=0)
      
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', transformer),
        ('svd',svd),
#         ('model',lda),
        ('clf', clf)
    ])
    #Simple Pipeline Fit
    pipeline.fit(X_train,Y_train)
    #Predict the train set
    predicted=pipeline.predict(X_train)
    print(metrics.classification_report(le.inverse_transform(Y_train), le.inverse_transform(predicted)))
    
    X = X_train
    Y = Y_train
    cv = StratifiedKFold(Y, n_folds=10)
    for i, (train, test) in enumerate(cv):
             
        print 'Calculating RandomForestClassifier...'
        pipeline.fit(X[train],Y[train])
        predicted = pipeline.predict(X[test])
        print "accuracy\t" + str(accuracy_score(Y[test],predicted))
        print "Finished calculation RandomForestClassifier!"
            
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    