import pandas as pd
from sklearn import preprocessing
from naiveBayes import NB
from knn import KNN
from svc import SVC
from randomForest import RF
import csv
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics.classification import accuracy_score


with open("./EvaluationMetric_10fold.csv", 'wb') as csvfile:
    writer = csv.writer(csvfile,delimiter=' ',quoting=csv.QUOTE_MINIMAL)
    writer.writerow('Statistic Measure\tNaive Bayes\tKNN\tSVM\tRandom Forest\tMy Method')
    accuracyKNeighborsClassifier = 0
    accuracySVC = 0
    accuracyRandomForestClassifier = 0
    accuracyMultinomialNB = 0
    
    df=pd.read_csv("train_set.csv",sep="\t")

    Y = df["Category"]
    le = preprocessing.LabelEncoder()
    le.fit(Y)
    Y=le.transform(Y)
    X = df['Content']
    Title = df['Title']
    
    nb = NB()
    k = KNN()
    s = SVC()
    r = RF()
    
    myAccuracy = []

    myPrediction = []
    j=0

    bestAccuracy = 0.0

    cv = StratifiedKFold(Y, n_folds=10)
    for i, (train, test) in enumerate(cv):
        
        nb.goWithInput(X,Y,Title, train, test)
        nbPredicted = le.inverse_transform(nb.getPredicted())
        accuracyMultinomialNB = nb.get_accuracy()
         
        k.goWithInput(X, Y, train, test)
        kPredicted = le.inverse_transform(k.getPredicted())
        accuracyKNeighborsClassifier = k.get_accuracy()
          
        s.goWithInput(X, Y , train, test)
        sPredicted = le.inverse_transform(s.getPredicted())
        accuracySVC = s.get_accuracy()
        
        r.goWithInput(X, Y , train, test)
        rPredicted = le.inverse_transform(r.getPredicted())
        accuracyRandomForestClassifier = r.get_accuracy()
        
        finalPrediction = []
        real = le.inverse_transform(Y[test])
        i = 0
        while(i < len(Y[test]) ):
            if(sPredicted[i] == nbPredicted[i] or sPredicted[i] == kPredicted[i] or sPredicted[i] == rPredicted[i] ):
                finalPrediction.append(sPredicted[i])
            elif(nbPredicted[i] == kPredicted[i] or nbPredicted[i] == rPredicted[i]):
                finalPrediction.append(kPredicted[i])
            else:
                finalPrediction.append(rPredicted[i])
            
            if( finalPrediction[i] != real[i] ):
                print nbPredicted[i] , ' ', kPredicted[i] , ' ',sPredicted[i] , ' ', rPredicted[i]
                print finalPrediction[i] , ' pragmatiko = ' , real[i]
            
            i+=1
         
        accuracy = accuracy_score(le.inverse_transform(Y[test]),finalPrediction)
        
        myAccuracy.append(accuracy)
        
        if(bestAccuracy< accuracy):
            myPrediction = finalPrediction
            bestAccuracy = accuracy
               
        print "Final accuracy = " , accuracy
        print accuracyMultinomialNB , ' ' , accuracyKNeighborsClassifier , ' ' , accuracySVC , ' ' ,accuracyRandomForestClassifier
     
    i=0
    while i < 10:
        writer.writerow('accuracy\t' + str(accuracyMultinomialNB[i]) + '\t' + str(accuracyKNeighborsClassifier[i]) + '\t'+str(accuracySVC[i])+'\t'+str(accuracyRandomForestClassifier[i]) + '\t'+ str(myAccuracy[i]) )
        i+=1