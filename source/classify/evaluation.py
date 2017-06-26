import numpy as np
from sklearn import svm

def randPerm(S,L,fact=0.8):
        indices = np.random.permutation(S.shape[0])
        training_idx, test_idx = indices[:int(l*fact)], indices[int(l*fact):]
        training, test = S[training_idx,:], S[test_idx,:]
        training_labels, test_labels = L[training_idx], L[test_idx]
        return training, test, training_labels, test_labels

class Evaluator: 
    
    def __init__(self,classifier):
        self._Classifier = classifier
    
    # checks only accuracy F-measure and other goodies must be added
    def single(self,training,training_labels,testing,testing_labels):
        self._Classifier.learn_mat(training,training_labels)
        return self._Classifier.test(testing,testing_labels)

    # must become more general
    # add an extension to SVM
    # to support multiple categories
    def kfold(self,S,L,k,fact =0.8,verbose = True):
        S1,S2 = S
        L1,L2 = L
        acc = 0
        for i in range(1, k+1):
            if verbose:
                print "Classification round: "+str(i)
            tr1, te1, trl1, tel1 = randPerm(S1,L1,fact=fact)
            tr2, te2, trl2, tel2 = randPerm(S2,L2,fact=fact)    
            training = np.concatenate((tr1, tr2), axis=0)
            training_labels = np.concatenate((trl1, trl2), axis=0)
            testing = np.concatenate((te1, te2), axis=0)
            testing_labels = np.concatenate((tel1, tel2), axis=0)
            acc+= self.single(training,training_labels,testing,testing_labels)
        return acc/k

