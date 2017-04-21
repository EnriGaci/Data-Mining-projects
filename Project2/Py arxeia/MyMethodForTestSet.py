from naiveBayes import NB
from knn import KNN
from svc import SVC
from randomForest import RF
import csv
import pandas as pd


with open("./testSet_categories.csv", 'wb') as csvfile:
    writer = csv.writer(csvfile,delimiter=' ',quoting=csv.QUOTE_MINIMAL)
    writer.writerow('ID\tPredicted Category')
    
    df = pd.read_csv("test_set.csv",sep="\t")
    id = df['Id']
    
    nb = NB()
    k = KNN()
    s = SVC()
    r = RF()

    nbPredicted = nb.goForTestCsv()
#     print nbPredicted 
            
    kPredicted = k.goForTestCsv()
#     print kPredicted
           
    sPredicted = s.goForTestCsv()
#     print sPredicted
          
    rPredicted = r.goForTestCsv()
#     print rPredicted
         
    finalPrediction = []
    i = 0
    while(i < len(nbPredicted) ):
        if(sPredicted[i] == nbPredicted[i] or sPredicted[i] == kPredicted[i] or sPredicted[i] == rPredicted[i] ):
            finalPrediction.append(sPredicted[i])
        elif(nbPredicted[i] == kPredicted[i] or nbPredicted[i] == rPredicted[i]):
            finalPrediction.append(kPredicted[i])
        else:
            finalPrediction.append(rPredicted[i])
        i+=1
    
    i=0
    while(i<len(finalPrediction)):
        writer.writerow(str(id[i]) + '\t' + str(finalPrediction[i]))
        i+=1
 
#      
#     i=0
#     while i < 10:
#         writer.writerow('accuracy\t' + str(accuracyMultinomialNB[i]) + '\t' + str(accuracyKNeighborsClassifier[i]) + '\t'+str(accuracySVC[i])+'\t'+str(accuracyRandomForestClassifier[i]) + '\t'+ str(myAccuracy[i]) )
#         i+=1