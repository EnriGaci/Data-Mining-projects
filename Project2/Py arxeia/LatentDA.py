from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pandas as pd
import csv
from numpy import empty

from gensim.utils import SaveLoad
from gensim.interfaces import TransformedCorpus

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')
en_stop+=["i","the","to","a","will","said","just","can","us","new","like","on","also",".",",","\u2013","and"]

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    

df=pd.read_csv("test_set.csv",sep="\t",encoding='utf-8')
doc_set = df['Content']

# list for tokenized documents in loop
texts = []

j=1
# loop through document list
for i in doc_set:
    
    if( j% 400 == 0):
        print "iter = ", j
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    
    # add tokens to list
    texts.append(stopped_tokens)

    j+=1

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=1000, id2word = dictionary)
# model =  models.LdaModel.load('lda.model')
# print model.show_topics(num_topics=10, num_words=10)

# topWords = []
# for t in range(0, model.num_topics-1):
# #     print 'topic {}: '.format(t) + ', '.join(str(i) for i in [v[0] for v in model.show_topic(t, 20)])
#     for word in [str(i) for i in [v[0] for v in model.show_topic(t, 20)]]:
#         topWords.append(word)
#     print "-------"


i=0
topicNumbers = []
with open("./testLdaWords_1000.csv", 'wb') as csvfile:
    csvfile.write('words\n')
    while(i<len(doc_set)):
#     while(i<10):
        probas = ldamodel.get_document_topics(corpus[i], minimum_probability=None)
#         probas = model.get_document_topics(corpus[i], minimum_probability=None)
        
#         print probas
        if not probas:
            topicNumbers.append('-1')
            i+=1
            continue
            print "empty probas"
            print corpus[i]
            print i
        maxProba = probas[0][1];
        maxProbaPos = 0;
        
        j=0
        for probability in probas:
            if(probability[1]> maxProba):
                maxProba = probability[1]
                maxProbaPos = j
            j+=1    
        
        topWords = []
        # to v[0] exei th leksh pou exei to topic me th megalyterh pi8anothta
#         for v in model.show_topic(probas[maxProbaPos][0], 20):
        for v in ldamodel.show_topic(probas[maxProbaPos][0], 20):
            topWords.append(v[0])
            csvfile.write(v[0].encode('utf8')+' ')
        csvfile.write('\n')
        
        topicNumbers.append(str(probas[maxProbaPos][0]))
        i+=1
        
        if( i % 200 == 0):
            print i
    
    print 5
      
# raw_data = {'topicNumber':topicNumbers}
# df2 = pd.DataFrame.from_dict(raw_data)   
# df2.to_csv('ldaWords_10.csv',sep="\t", encoding='utf-8')




# raw_data={'LDA_10':ldamodel.print_topics(num_topics=10,num_words=10)}
# df2 = pd.DataFrame(raw_data, columns = ['LDA_10'])
# df2.to_csv('lda_10.csv',sep="\t")

# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word = dictionary)

# raw_data={'LDA_100':ldamodel.print_topics(num_topics=10,num_words=10)}
# df2 = pd.DataFrame(raw_data, columns = ['LDA_100'])
# df2.to_csv('lda_100.csv',sep="\t")

# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=1000, id2word = dictionary)

# raw_data={'LDA_1000':ldamodel.print_topics(num_topics=10,num_words=10)}
# df2 = pd.DataFrame(raw_data, columns = ['LDA_1000'])
# df2.to_csv('lda_1000.csv',sep="\t")