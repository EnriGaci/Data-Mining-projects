from naiveBayes import NB
from knn import KNN
from svc import SVC
from randomForest import RF
import csv

with open("./EvaluationMetric_10fold.csv", 'wb') as csvfile:
    writer = csv.writer(csvfile,delimiter=' ',quoting=csv.QUOTE_MINIMAL)
        
    nb = NB()
    nb.go()
    accuracyMultinomialNB = nb.get_accuracy()
    print accuracyMultinomialNB
           
    k = KNN()
    k.go()
    accuracyKNeighborsClassifier = k.get_accuracy()
          
    s = SVC()
    s.go()
    accuracySVC = s.get_accuracy()
           
    r = RF()
    r.go()
    accuracyRandomForestClassifier = r.get_accuracy()
       
    i=0
    while i < 10:
        writer.writerow('accuracy\t' + str(accuracyMultinomialNB[i]) + '\t' + str(accuracyKNeighborsClassifier[i]) + '\t'+str(accuracySVC[i])+'\t'+str(accuracyRandomForestClassifier[i]) )
        i+=1