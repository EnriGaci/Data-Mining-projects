from sklearn.metrics.classification import accuracy_score
import matplotlib.pyplot as plt

class Util:
    
    def calcRoc(self,le,Y_train,test,predicted):
            
        actual = le.inverse_transform(Y_train[test])
        predic = le.inverse_transform(predicted)
         
        politicsFp = 0
        politicsTp = 0
        filmFp = 0
        filmTp = 0
        footballFp = 0
        footballTp = 0
        businessFp = 0
        businessTp = 0
        technologyFp = 0
        technologyTp = 0
         
        politicsFpr = [0.0]
        politicsTpr = [0.0]
        filmTpr = []
        filmFpr = []
        footballFpr = []
        footballTpr = []
        businessFpr = []
        businessTpr = []
        technologyFpr = []
        technologyTpr = []
         
        politicsCounter = 0
        filmCounter = 0
        footballCounter = 0
        businessCounter = 0
        technologyCounter = 0
        i = 0 
        while i < len(Y_train[test]):
            
            if(actual[i] == 'Politics'):
                politicsCounter += 1
            elif(actual[i] == 'Film'):
                filmCounter +=1
            elif(actual[i] == 'Football'):
                footballCounter+=1
            elif(actual[i] == 'Business'):
                businessCounter+=1
            else:
                technologyCounter+=1
                 
            if( predic[i] == 'Politics' ):
                if(actual[i] == 'Politics'):
                    politicsTp += 1
                else:
                    politicsFp += 1
            elif(predic[i] == 'Film'):
                if(actual[i] == 'Film'):
                    filmTp += 1
                else:
                    filmFp += 1
            elif(predic[i] == 'Football'):
                if(actual[i] == 'Football'):
                    footballTp += 1
                else:
                    footballFp += 1
            elif(predic[i] == 'Business'):
                if(actual[i] == 'Business'):
                    businessTp += 1
                else:
                    businessFp += 1
            else:
                if(actual[i] == 'Technology'):
                    technologyTp += 1
                else:
                    technologyFp += 1
            
            #         politicsFpr.append(float(politicsFp)/9583)
            #         politicsTpr.append(float(politicsTp)/2683)
            
            if( i % 500 == 0): 
                if(politicsCounter!= 0 ):
                    if(i-politicsCounter != 0):
                        politicsFpr.append(float(politicsFp)/(i-politicsCounter))
                    else:
                        politicsFpr.append(0.0)
                    politicsTpr.append(float(politicsTp)/politicsCounter)
                else:
                    politicsFpr.append(0.0)
                    politicsTpr.append(0.0)
                  
                #         filmFpr.append(float(filmFp)/10026)
                #         filmTpr.append(float(filmTp)/2240)
                 
                if(filmCounter!= 0 ):
                    if(i-filmCounter != 0):
                        filmFpr.append(float(filmFp)/(i-filmCounter))
                    else:
                        filmFpr.append(0.0)
                    filmTpr.append(float(filmTp)/filmCounter)
                else:
                    filmFpr.append(0.0)
                    filmTpr.append(0.0)
                  
                #         footballFpr.append(float(footballFp)/9145)
                #         footballTpr.append(float(footballTp)/3121)
                 
                if(footballCounter!= 0 ):
                    if(i-footballCounter != 0):
                        footballFpr.append(float(footballFp)/(i-footballCounter))
                    else:
                        footballFpr.append(0.0)
                    footballTpr.append(float(footballTp)/footballCounter)
                else:
                    footballFpr.append(0.0)
                    footballTpr.append(0.0)
                  
                #         businessFpr.append(float(businessFp)/9531)
                #         businessTpr.append(float(businessTp)/2735)
                 
                if( businessCounter!= 0 ):
                    if(i-businessCounter != 0):
                        businessFpr.append(float(businessFp)/(i-businessCounter))
                    else:
                        businessFpr.append(0.0)
                    businessTpr.append(float(businessTp)/businessCounter)
                else:
                    businessFpr.append(0.0)
                    businessTpr.append(0.0)
                  
                #         technologyFpr.append(float(technologyFp)/10779)
                #         technologyTpr.append(float(technologyTp)/1487)
                 
                if(technologyCounter!= 0 ):
                    if(i-technologyCounter != 0):
                        technologyFpr.append(float(technologyFp)/(i-technologyCounter))
                    else:
                        technologyFpr.append(0.0)
                    technologyTpr.append(float(technologyTp)/technologyCounter)
                else:
                    technologyFpr.append(0.0)
                    technologyTpr.append(0.0)
                  
            i+=1
        
        print "\tPolitics\tFilm\tBusiness\tTechnology\tFootball"
        print "Fpr\t"+str(politicsFpr[-1])+'\t'+str(filmFpr[-1])+'\t'+str(businessFpr[-1])+'\t'+str(technologyFpr[-1])+'\t'+str(footballFpr[-1])+'\t'
        print "Tpr\t"+str(politicsTpr[-1])+'\t'+str(filmTpr[-1])+'\t'+str(businessTpr[-1])+'\t'+str(technologyTpr[-1])+'\t'+str(footballTpr[-1])+'\t'
        
        plt.xlabel("FPR", fontsize=14)
        plt.ylabel("TPR", fontsize=14)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle='dashed', color='red', linewidth=1)
        plt.plot(politicsFpr,politicsTpr,label = 'Politics')
        plt.plot(filmFpr,filmTpr,label='Film')
        plt.plot(footballFpr,footballTpr,label = 'Football')
        plt.plot(businessFpr,businessTpr,label = 'Business')
        plt.plot(technologyFpr,technologyTpr,label = 'Technology')
        plt.legend(loc='best')
        plt.show() 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        