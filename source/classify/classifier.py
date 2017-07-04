import numpy as np
from sklearn import svm


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

    def kernelization(self,X,t):
        # for 1 to 3 array is considered symmetric
        if(t==1):
            #spectrum clip
            e,v = np.linalg.eig(X)
            ep = np.maximum.reduce([e,np.zeros(e.shape[0])])
            S = np.dot(v.T,np.dot(numpy.diag(ep),v))
        elif(t==2):
            #spectrum flip
            e,v = np.linalg.eig(X)
            ep = np.abs(e)
            S = np.dot(v.T,np.dot(numpy.diag(ep),v))
        elif(t==3):
            #spectrum shift
            e,v = np.linalg.eig(X)
            minS = np.min(e)
            minS = min(minS,0)
            if (minS==0):
                S = X
            else:
                S = np.dot(v.T,np.dot(numpy.diag(e+minS),v))
        else:
            #spectrum square
            S = np.dot(X,X.T)
        return S
            
    def learn_mat(self,X,labels,t = 4):
        self._clf = svm.SVC(kernel='precomputed')
        kernel = self.kernelization(X,t)
        self._clf.fit(kernel,labels)
    
    def classify(self,X):
    	return self._clf.predict(X)
