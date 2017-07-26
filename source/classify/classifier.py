import numpy as np
from sklearn import svm

def kernelization(X,t=0):
        # for 1 to 3 array is considered symmetric
        if(t==1):
            #spectrum clip
            e,v = np.linalg.eig(X)
            ep = np.maximum.reduce([e,np.zeros(e.shape[0])])
            S = np.dot(v.T,np.dot(np.diag(ep),v))
        elif(t==2):
            #spectrum flip
            e,v = np.linalg.eig(X)
            ep = np.abs(e)
            S = np.dot(v.T,np.dot(np.diag(ep),v))
        elif(t==3):
            #spectrum shift
            e,v = np.linalg.eig(X)
            minS = np.min(e)
            minS = min(minS,0)
            if (minS==0):
                S = X
            else:
                S = np.dot(v.T,np.dot(np.diag(e+minS),v))
        elif(t==4):
            #spectrum square
            S = np.dot(X,X.T)
	else:
            #leave as is
            S = X
        return S

class classifier: 
    
    def __init__(self):
        pass
    def learn(self):
        pass
    def classify(self):
        pass
    
class SVM(classifier):
    def __init__(self):
        self._clf = None 

    def learn_mat(self,X,labels,propability=False):
	# input is in the form of a valid kernel
        self._clf = svm.SVC(kernel='precomputed',propability = propability)
        self._clf.fit(X,labels)
        
    def decision_function(self,X_test):
        return self._clf.decision_function(X_test)
        
    def classify(self,X):
    	return self._clf.predict(X)

    def getClassifier(self):
        return self._clf