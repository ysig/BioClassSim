import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

def randPerm(S,L,fact=0.8):
        ltotal = 0
        for i in range(len(S)):
            l = S[i].shape[0]
            indices = np.random.permutation(l)+ltotal
            limit = int(l*fact)
            if(i==0):
                training_idx = indices[:limit]
                testing_idx = indices[limit:]
                Similarity = S[i]
                Labels = L[i]
            else:
                training_idx = np.append(training_idx,indices[:limit],axis=0)
                testing_idx = np.append(testing_idx,indices[limit:],axis=0)
                Similarity = np.append(Similarity, S[i], axis=0)
                Labels = np.append(Labels, L[i], axis=0)
            ltotal+=l
        training = Similarity[training_idx,:]
        training = training[:,training_idx]
        
        testing = Similarity[testing_idx,:]
        testing = testing[:,training_idx]

        training_labels = Labels[training_idx]
        testing_labels = Labels[testing_idx]
        
        return training,training_labels,testing,testing_labels

def calculateTFNP(cm):
    diag = np.diag(cm)
    FP = np.sum(np.sum(cm,axis=0) - diag)
    FN = np.sum(np.sum(cm,axis=1) - diag)
    TP = np.sum(diag)
    TN = np.sum(cm) - (FP + FN + TP)
    return FP,FN,TP,TN

def CalculateMetrics(cm):
    FP,FN,TP,TN = calculateTFNP(cm)
    metrics = dict()
    # Sensitivity, hit rate, recall, or true positive rate
    metrics['recall'] = TP/(TP+FN)
    # Specificity or true negative rate
    metrics['specifity'] = TN/(TN+FP) 
    # Precision or positive predictive value
    metrics['precision'] = TP/(TP+FP)
    # Negative predictive value
    metrics['negative_predictive_value'] = TN/(TN+FN)
    # Fall out or false positive rate
    metrics['fall_out'] = FP/(FP+TN)
    # False negative rate
    metrics['false_negative_rate'] = FN/(TP+FN)
    # False discovery rate
    metrics['false_discovery_rate'] = FP/(TP+FP)

    # Overall accuracy
    metrics['accuracy'] = (TP+TN)/(TP+FP+FN+TN)
    return metrics

def displayMetrics(metrics):
    print "Sensitivity: ",metrics['recall']
    print "Specifity: ",metrics['specifity']
    print "Precision: ",metrics['precision']
    print "Negative predictive value: ", metrics['negative_predictive_value']
    print "Fall out: ",metrics['fall_out']
    print "False negative rate: ", metrics['false_negative_rate']
    print "False discovery rate: ", metrics['false_discovery_rate']
    print "Accuracy: ", metrics['accuracy']

class Evaluator: 
    
    def __init__(self,classifier):
        self._Classifier = classifier
    
    def single(self,training,training_labels,testing,testing_labels,calculate_metrics = True):
        self._Classifier.learn_mat(training,training_labels)
        Lp = self._Classifier.classify(testing)
        cm = confusion_matrix(testing_labels, Lp)
        if (calculate_metrics==True):
            return CalculateMetrics(cm),cm
        else:
            return cm

    def kfold(self,S,L,k,fact =0.8,verbose = False):
        for i in range(1, k+1):
            if verbose:
                print "Classification round: "+str(i)
            training,training_labels,testing,testing_labels = randPerm(S,L,fact=fact)
            if(i==1):
                cm = self.single(training,training_labels,testing,testing_labels,calculate_metrics = False)
            else: 
                cm += self.single(training,training_labels,testing,testing_labels,calculate_metrics = False)
        metrics = CalculateMetrics(cm)
        if verbose:
            print "Displaying Metrics.."
            displayMetrics(metrics)
        return metrics, cm
