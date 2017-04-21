#!/usr/bin/python

import os,re
import numpy as np
import math
import csv
import sys
import codecs
import scipy as sp
from pandas import Series, DataFrame
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from stop_words import get_stop_words
from nltk.corpus import stopwords
from scipy import spatial
from six import iteritems

sys.stdout = codecs.getwriter("iso-8859-1")(sys.stdout, 'xmlcharrefreplace')

NUMBER_OF_CLUSTER=5
MAX_ITERATIONS=100

def kmeans_plus(points, number_of_cluster , Y_train):

    # initialize and randomize cendroids in the first step
    centroids =[]
    centroids = initialize(points,number_of_cluster)
    # for cluster in range(0, number_of_cluster):
    #     centroids.append(points[np.random.randint(0, len(points), size=1)].flatten().tolist())
    #print centroids
    #keep old centroids to know when the algorithm terminate
    old_centroids = [[] for i in range(number_of_cluster)] 
    #we have to write the results of algorithm
    dataCsv=[]
    dataCsv=[['Business','Films','Football','Politics','Technology']]

    #iterations of algorithm
    iterations = 0

    #we stop if we not found new centroids 
    while not (old_centroids == centroids or iterations==MAX_ITERATIONS):

        iterations += 1
        print(iterations)

        #initialize clusters
        clusters = [[] for i in range(number_of_cluster)]

        #we need to find real categories for predict accuracy of algorithm
        index_of_category = [[] for i in range(number_of_cluster)]

        #add point to corect clusters  
        clusters = add_points_to_cluster(points, centroids, clusters,index_of_category)

        #now we have to calc new centroids with new points
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = calc_new_cendroids(cluster,centroids[index])
            index += 1

    #calc the results of kmeans
    cluster_counter = 0
    for index in index_of_category:

        results=[0,0,0,0,0]

        count0=0
        count1=0
        count2=0
        count3=0
        count4=0
        count=0
        for x in index:
            if Y_train[x]==0:
                count0+=1
            elif Y_train[x]==1:
                count1+=1
            elif Y_train[x]==2:
                count2+=1
            elif Y_train[x]==3:
                count3+=1
            else:
                count4+=1
            count+=1
        print "GIA TO CLUSTER # " ,cluster_counter," TA POSOSTA GIA KATHE KATHGORIA EINAI:"
        results[0]=float(count0)/float((sum(Y_train == 0)))
        print "Business = " ,float(count0)/float((sum(Y_train == 0))),"  " ,count0,"/" ,sum(Y_train == 0)
        results[1]=float(count1)/float((sum(Y_train == 1)))
        print "Film = " ,float(count1)/float((sum(Y_train == 1))),"  " ,count1,"/" ,sum(Y_train == 1)
        results[2]=float(count2)/float((sum(Y_train == 2)))
        print "Football = " ,float(count2)/float((sum(Y_train == 2))),"  " ,count2,"/" ,sum(Y_train == 2)
        results[3]=float(count3)/float((sum(Y_train == 3)))
        print "Politics = " ,float(count3)/float((sum(Y_train == 3))),"  " ,count3,"/" ,sum(Y_train == 3)
        results[4]=float(count4)/float((sum(Y_train == 4)))
        print "Tecnology = " , float(count4)/float((sum(Y_train == 4))),"  ",count4,"/",sum(Y_train == 4)
        print "sunolika # " ,count," documents"
        cluster_counter+=1
        print "-----------------------------------------------------------------------------------"

        dataCsv.append(results)

    print "The total number of iterations necessary is: " ,iterations
    
    #write the results in csv file
    with open('clustering_KMeans.csv','wb') as fp:
        a=csv.writer(fp,delimiter=',')
        a.writerows(dataCsv)
    return

      
def add_points_to_cluster(points, centroids, clusters,index_of_category):
    count=0
    for instance in points:  
        
        #calc distance by cosine similarity and put them to centroids with max similarity
        # mu_index = max([(i[0],cosine_similarity(instance, centroids[i[0]])) for i in enumerate(centroids)], key=lambda t:t[1])[0]
        mu_index = min([(i[0],spatial.distance.cosine(instance, centroids[i[0]])) for i in enumerate(centroids)], key=lambda t:t[1])[0]
        try:
            clusters[mu_index].append(instance)
            index_of_category[mu_index].append(count)
        except KeyError:
            clusters[mu_index] = [instance]
            index_of_category[mu_index]=[count]
        count+=1
    #check the contition if a cluster is empty
    for cluster in clusters:
        if not cluster:
            cluster.append(points[np.random.randint(0, len(points), size=1)].flatten().tolist())
    return clusters

def cosine_similarity(inst,cen):
    ar=np.array(cen)
    sum1=0
    q1=inst*ar
    for element in q1:
        sum1+=element
    sum2=0
    q2=inst*inst
    for element in q2:
        sum2+=element
    sum3=0
    q3=ar*ar
    for element in q3:
        sum3+=element

    if (sum1 == 0 and (sum2 == 0 or sum3 == 0)):
        similarity = 0
    else:
        similarity = sum1/(math.sqrt(sum2)*math.sqrt(sum3))
    #print("similarity = "+str(similarity))
    return similarity

#we calc the average of points
def calc_new_cendroids(cluster,centroid):
    
    sumOfPoints=[]
    for i in range(len(cluster[0])):
        sumOfPoints.append(0)
    for points in cluster:
        j=0
        for x in points:
            sumOfPoints[j]+=x
            j+=1
    centroid= [x / len(cluster[0]) for x in sumOfPoints]
    return centroid


#initialize first 5 centroids(kmeans++)
def initialize(X,K):
	C=[]
	C.append([X[0]])
	for k in range(1,K):
		D = np.array([min([np.inner(np.array(c)-np.array(x),np.array(c)-np.array(x)) for c in C]) for x in X])
		props=D/float(D.sum())
		SumCum=props.cumsum()
		r=sp.rand()
		for j,p in enumerate(SumCum):
			if r<p:
				break
		C.append(X[j])
	return C

#preprocessing

#gathering data
# df=pd.read_csv("newTrain_set.csv",sep="\t")
df=pd.read_csv("train_set.csv",sep="\t")
X_train = df["Content"]
Z_train = df["Title"]
#prepocessing Category
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train=le.transform(df["Category"])

#cleaning data
#stop words
stopWords = get_stop_words('english')
stopWords+=['said', 'will', 'know','want', 'say', 'also', 'good', 'well', 'made', 'year', 'new', 'next', 'going', 'high','best','think','play','one','much','even','come'] # remove it if you need punctuation 
stopWords+=['day','last','since','take','now','much','still','make','maybe','thing','see','lot','even','way','told','long','put']

#put title to content
X_train+=5*Z_train.copy()

#initialize CounterVector
vectorizer=CountVectorizer(stop_words=stopWords,max_df=0.85,min_df=0.0005)

transformer=TfidfTransformer()
tfidf=TfidfVectorizer(stop_words=stopWords,max_df=0.2,min_df=1)

svd=TruncatedSVD(n_components=100, random_state=42)

# counter vectors for content
CounterVectorContent=vectorizer.fit_transform(X_train)
# tf=tfidf.fit_transform(X_train)

#normalitation data
FrequencyCounterVectorContent=transformer.fit_transform(CounterVectorContent)

#lsi
lsiC=svd.fit_transform(FrequencyCounterVectorContent)
kmeans_plus(lsiC,NUMBER_OF_CLUSTER,Y_train)
