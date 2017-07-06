import numpy as np
import os
import sys
#import class files
sys.path.append('..')
from source import bioRead as br
from source import classify as cl
#import PyInsect for measuring similarity
sys.path.append('../../')
from PyINSECT import representations as REP
from PyINSECT import comparators as CMP


npz = np.load('SimilaritiesAndDictionaries/metrics.npz')
cm = npz['cm']
metrics = npz['metrics']

print cm
FP,FN,TP,TN = cl.calculateTFNP(cm)
print FP
print FN
print TP
print TN
