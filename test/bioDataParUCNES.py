import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import itertools
import os
import sys
import dill
#import class files
sys.path.append('..')
from source import bioRead as br
from source import classify as cl
#import PyInsect for measuring similarity
sys.path.append('../../')
from PyINSECT import representations as REP
from PyINSECT import comparators as CMP

def represent(a):
    ngg = REP.DocumentNGramGraph(3,2,a)
    return ngg

def SOP(ngg1,ngg2,sop):
	return sop(ngg1,ngg2)

if os.path.exists('SimilaritiesAndDictionaries/UCNE.npz'):

    npz = np.load('SimilaritiesAndDictionaries/UCNE.npz')
    hd = npz['hd']
    cd = npz['cd']
    S = npz['S']
    l1 = npz['l1']
    l2 = npz['l2']
    l = npz['l']
    L = np.append(np.zeros(l1),np.ones(l2),axis=0)
else:
    sr = br.SequenceReader()
    
    sr.read('./biodata/UCNEs/hg19_UCNEs.fasta')
    hd = sr.getDictionary()

    print "Gained Human Dictionary"

    sr.read('./biodata/UCNEs/galGal3_UCNEs.fasta')
    cd = sr.getDictionary()

    print "Gained Chicken Dictionary"

    n=3
    Dwin=2
    subjectMap = {}
    ngg = {}
    l1 = len(hd.keys())
    l2 = len(cd.keys())
    l = l1 + l2

    print "Size(hdict)="+str(l1)
    print "Size(cdict)="+str(l2)

    i = 0
	
    num_cores = multiprocessing.cpu_count()-1

    ngg = Parallel(n_jobs=num_cores)(delayed(represent)(a) for key,a in itertools.chain(hd.iteritems(),cd.iteritems()))
    print "Graphs Created [Humans,Chickens]"
    
    subjectMap = zip(itertools.chain(hd.iterkeys(),cd.iterkeys()),l1*['humans']+l2*['chickens'])    
    
    S = np.empty([l, l])
    L = np.empty([l])

    sop = []
    for i in range(0,num_cores):
        sop.append(CMP.SimilarityNVS())

    for i in range(0,l1):
        print i," ",
        L[i] = 0 #0 for humans
        S[i:,i] = Parallel(n_jobs=num_cores)(delayed(SOP)(ngg[i],ngg[j],sop[j % num_cores].apply) for j in range(i,l))

    print ""

    for i in range(l1,l):
        print i," ",
        L[i] = 1 #1 for chickens
        S[i:,i] = Parallel(n_jobs=num_cores)(delayed(SOP)(ngg[i],ngg[j],sop[j % num_cores].apply) for j in range(i,l))
		
    print ""

    for i in range(0,l):
        for j in range(0,i):
            S[i,j] = S[j,i]

    print "Similarity matrix constructed.."

    if not os.path.exists('SimilaritiesAndDictionaries'):
        os.mkdir('SimilaritiesAndDictionaries')

    np.savez('SimilaritiesAndDictionaries/UCNE.npz', hd=hd, cd=cd, l1=l1, l2=l2, l=l, S=S)

evaluator = cl.Evaluator(cl.SVM())
reps = 10

S1 = S[0:l1,:]
S2 = S[l1:,:]
L1 = L[0:l1]
L2 = L[l1:]

metrics = evaluator.kfold((S1,S2),(L1,L2),reps,verbose=True)
np.savez('SimilaritiesAndDictionaries/metrics.npz', metrics=metrics)

