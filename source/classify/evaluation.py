import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, roc_auc_score

# randomly permutes in equal amount and segments
# Samples,labels by a factor of "fact"
# Similarity Touple:
# (S1,S2,...,Sk) where k the number of categories
# and S1.size = n1 x (n1 + ... + nk)
#     S2.size = n2 x (n1 + ... + nk)
#        .      .  .        . 
#        .      .  .        . 
#        .      .  .        . 
#     Sk.size = nk x (n1 + ... + nk)
# where ni the number of samples in the ith category
# concat(S1,...,Sk) = SimilarityMatrix
#
# Labels Touple:
# (L1, ... , Lk): where Li a ni x 1 matrix with all elements
# equal to the category number corresponding to Si
def randPerm(Similarity_Touple,LabelesTouple,fact=0.8):
        ltotal = 0
        for i in range(len(Similarity_Touple)):
            l = Similarity_Touple[i].shape[0]
            # indices are taken randomly
            # based on the sample size and append 
            # by a total length
            indices = np.random.permutation(l)+ltotal
            limit = int(l*fact)
            if(i==0):
                training_idx = indices[:limit]
                testing_idx = indices[limit:]
                Similarity = Similarity_Touple[i]
                Labels = LabelesTouple[i]
            else:
                training_idx = np.append(training_idx,indices[:limit],axis=0)
                testing_idx = np.append(testing_idx,indices[limit:],axis=0)
                Similarity = np.append(Similarity, Similarity_Touple[i], axis=0)
                Labels = np.append(Labels, LabelesTouple[i], axis=0)
            ltotal+=l
        training = Similarity[training_idx,:]
        training = training[:,training_idx]
        
        testing = Similarity[testing_idx,:]
        testing = testing[:,training_idx]

        training_labels = Labels[training_idx]
        testing_labels = Labels[testing_idx]
        
        return training,training_labels,testing,testing_labels

# Calculate TP,FP,TN,FN
def calculateTFNP(ConfusionMatrix):
    s = np.sum(ConfusionMatrix)
    TP = np.zeros(ConfusionMatrix.shape[0])
    FP = np.zeros(ConfusionMatrix.shape[0])
    FN = np.zeros(ConfusionMatrix.shape[0])
    TN = np.zeros(ConfusionMatrix.shape[0])
    for i in range(ConfusionMatrix.shape[0]):
        TP[i] = ConfusionMatrix[i, i] 
        FP[i] = np.sum(ConfusionMatrix, axis=0)[i] - ConfusionMatrix[i, i]
        FN[i] = np.sum(ConfusionMatrix, axis=1)[i] - ConfusionMatrix[i, i]
        TN[i] = s - TP[i] - FP[i] - TN[i]
    return FP,FN,TP,TN

# has dummy indicates if class 0
# signifies a dummy (negative for all)
# class
def CalculateMetrics(ConfusionMatrix,has_dummy=False):
# considers no dummy class
    FP,FN,TP,TN = calculateTFNP(ConfusionMatrix)
    metrics = dict()
    
    if(has_dummy):
        TP = np.delete(TP,0,0)
        TN = np.delete(TN,0,0)
        FP = np.delete(FP,0,0)
        FN = np.delete(FN,0,0)

    # True Positive
    metrics['TP'] = TP
    # False Positive
    metrics['FP'] = FP
    # False Negative
    metrics['FN'] = FN
    # True Negative
    metrics['TN'] = TN
    
    # Microaverage metrics
    tmetrics = dict()    
    # Sensitivity, hit rate, recall, or true positive rate
    TPs = float(np.sum(TP,axis=0))
    FNs = float(np.sum(FN,axis=0))
    TNs = float(np.sum(TN,axis=0))
    FPs = float(np.sum(FP,axis=0))
    tmetrics['recall'] = np.divide(TPs,(TPs+FNs))
    # Specificity or true negative rate
    tmetrics['specifity'] = np.divide(TNs,(TNs+FPs))
    # Precision or positive predictive value
    tmetrics['precision'] = np.divide(TPs,(TPs+FPs))
    # Negative predictive value
    tmetrics['negative_predictive_value'] = np.divide(TNs,(TNs+FNs))
    # Fall out or false positive rate
    tmetrics['fall_out'] = np.divide(FPs,(FPs+TNs))
    # False negative rate
    tmetrics['false_negative_rate'] = np.divide(FNs,(TPs+FNs))
    # False discovery rate
    tmetrics['false_discovery_rate'] = np.divide(FPs,(TPs+FPs))
    # Fmeasure
    tmetrics['Fmeasure'] = np.divide(2*TPs,(2*TPs+FNs+FPs))
    # Overall accuracy
    tmetrics['accuracy'] = np.divide((TPs+TNs),(TPs+FPs+FNs+TNs))
    metrics['microaverage'] = tmetrics

    # Macroaverage metrics
    tmetrics = dict()
    # Sensitivity, hit rate, recall, or true positive rate
    tmetrics['recall'] = np.mean(np.divide(TP,1.0*np.add(TP,FN)))
    # Specificity or true negative rate
    tmetrics['specifity'] = np.mean(np.divide(TN,1.0*(np.add(TN,FP))))
    # Precision or positive predictive value
    tmetrics['precision'] = np.mean(np.divide(TP,1.0*(np.add(TP,FP))))
    # Negative predictive value
    tmetrics['negative_predictive_value'] = np.mean(np.divide(TN,1.0*(np.add(TN,FN))))
    # Fall out or false positive rate
    tmetrics['fall_out'] = np.mean(np.divide(FP,1.0*(np.add(FP,TN))))
    # False negative rate
    tmetrics['false_negative_rate'] = np.mean(np.divide(FN,1.0*(np.add(TP,FN))))
    # False discovery rate
    tmetrics['false_discovery_rate'] = np.mean(np.divide(FP,1.0*(np.add(TP,FP))))
    # Fmeasure
    tmetrics['Fmeasure'] = np.mean(np.divide(2*TP,1.0*(np.add(np.add(2*TP,FN),FP))))
    # Overall accuracy
    tmetrics['accuracy'] = np.mean(np.divide(np.add(TP,TN),np.add(np.add(np.add(TP,FP),FN),TN)))
    metrics['macroaverage'] = tmetrics
    return metrics

def displayMetrics(Metrics,case = 1):

    if(case<=1):
        metrics = Metrics['microaverage']
        print "Microaverage Metrics.. \n"
        print "Sensitivity: ",metrics['recall']
        print "Specifity: ",metrics['specifity']
        print "Precision: ",metrics['precision']
        print "Negative predictive value: ", metrics['negative_predictive_value']
        print "Fall out: ",metrics['fall_out']
        print "False negative rate: ", metrics['false_negative_rate']
        print "False discovery rate: ", metrics['false_discovery_rate']
        print "Fmeasure: ", metrics['Fmeasure']
        print "Accuracy: ", metrics['accuracy']

    if(case>=1):
        metrics = Metrics['macroaverage']
        print "\nMacroaverage Metrics.. \n"
        print "Sensitivity: ",metrics['recall']
        print "Specifity: ",metrics['specifity']
        print "Precision: ",metrics['precision']
        print "Negative predictive value: ", metrics['negative_predictive_value']
        print "Fall out: ",metrics['fall_out']
        print "False negative rate: ", metrics['false_negative_rate']
        print "False discovery rate: ", metrics['false_discovery_rate']
        print "Fmeasure: ", metrics['Fmeasure']
        print "Accuracy: ", metrics['accuracy']

class Evaluator: 
    
    def __init__(self,classifier):
        self._Classifier = classifier
    
    # calculates AUC
    def AUC(self,training,training_labels,testing,testing_labels):
        classifier = self._Classifier
        classifier.learn_mat(training,training_labels,probability = True)
        probabilities = classifier.predict_prob(testing)
        return roc_auc_score(testing_labels, probabilities[:,1])
        
    # single classification experiment
    # has dummy variable is given to calculate metric has dummy var
    def single(self,training,training_labels,testing,testing_labels,calculate_metrics = True, has_dummy = False):
        classifier = self._Classifier
        classifier.learn_mat(training,training_labels)
        Lp = classifier.classify(testing)
        ConfusionMatrix = confusion_matrix(testing_labels, Lp)
        if (calculate_metrics==True):
            return CalculateMetrics(ConfusionMatrix,has_dummy),ConfusionMatrix
        else:
            return ConfusionMatrix

    # conduct a randomized k-fold experiment
    # Verbose determines state printing
    # for Labeles Touple, Similarity Touple
    # see how they are defined on randPerm() -> goto page top
    def Randomized_kfold(self,SimilarityTouple,LabelesTouple,k,fact =0.8,verbose = False):
        for i in range(1, k+1):
            if verbose:
                print "Classification round: "+str(i)
            training,training_labels,testing,testing_labels = randPerm(SimilarityTouple,LabelesTouple,fact=fact)
            if(i==1):
                ConfusionMatrix = self.single(training,training_labels,testing,testing_labels,calculate_metrics = False)
            else: 
                ConfusionMatrix = np.add(ConfusionMatrix,self.single(training,training_labels,testing,testing_labels,calculate_metrics = False))
        metrics = CalculateMetrics(ConfusionMatrix)
        if verbose:
            print "Displaying Metrics.."
            displayMetrics(metrics)
        return metrics, ConfusionMatrix
