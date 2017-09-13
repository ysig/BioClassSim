import numpy as np
import os
import sys
import math
#import class files
# sys.path.append('../../../')
from source import bioRead as br
from source import classify as cl
#import PyInsect for measuring similarity
#sys.path.append('../../../../')
from PyINSECT import representations as REP
from PyINSECT import comparators as CMP
from multiprocessing import Pool
import multiprocessing

# Local function
def __getSimilaritiesForIndex(setting):
    i, l, S, ngg = setting # Explode
    for j in range(i,l):
        dTmp = sop.getSimilarityDouble(ngg[i],ngg[j])
        if (math.isnan(dTmp)):
            raise Exception("Invalid similarity! Check similarity implementation.")
        S[i,j] = dTmp
# End local function


# If we have cached the main analysis data
if os.path.exists('SimilaritiesAndDictionaries/UCNE.npz'):
    # Use them
    npz = np.load('SimilaritiesAndDictionaries/UCNE.npz')
    hd = npz['hd']
    cd = npz['cd']
    S = npz['S']
    l1 = npz['l1']
    l2 = npz['l2']
    l = npz['l']
    L = np.append(np.zeros(l1),np.ones(l2),axis=0)
    print "WARNING: Using cached data!"
else:
    # else start reading
    sr = br.SequenceReader()

    # Get Human UCNE fasta data
    sr.read('./biodata/UCNEs/hg19_UCNEs.fasta')
    # sr.read('./biodata/UCNEs/hg19_UCNEs-10.fasta')
    hd = sr.getDictionary()

    print "Gained Human Dictionary"

    # Get Chicken UCNE fasta data
    sr.read('./biodata/UCNEs/galGal3_UCNEs.fasta')
    # sr.read('./biodata/UCNEs/galGal3_UCNEs-10.fasta')
    cd = sr.getDictionary()

    print "Gained Chicken Dictionary"

    # Set n-gram graph analysis parameters
    n=3
    Dwin=2

    subjectMap = {}
    ngg = {}
    # Get number of UNCEs (for either type of UNCE)
    l1 = len(hd.keys())
    l2 = len(cd.keys())
    l = l1 + l2

    print "Found %d human UNCEs"%(l1)
    print "Found %d chicken UNCEs"%(l2)


    # For every human UNCE
    i = 0
    for key,a in hd.iteritems():
        # Assign appropriate label
        subjectMap[i] = (key,'humans')
        # Create corresponding graph
        ngg[i] = REP.DocumentNGramGraph(n,Dwin,a)
        i += 1

    print "Graphs Created for Humans"

    for key,b in cd.iteritems():
        subjectMap[i] = (key,'chickens')
        ngg[i] = REP.DocumentNGramGraph(n,Dwin,b)
        i += 1


    print "Graphs Created for Chickens"

    S = np.empty([l, l])
    L = np.empty([l])
    sop = CMP.SimilarityNVS()

    print "Getting human similarities..."
    # TODO: Examine default (problems with locking S)
    # pThreadPool = Pool(1);

    qToExecute = list() # Reset tasks
    for i in range(0,l1):
        print i," ",
        L[i] = 0 #0 for humans
        qToExecute.append((i,l,S,ngg))

    # pThreadPool.map(__getSimilaritiesForIndex, qToExecute)
    map(__getSimilaritiesForIndex,qToExecute)

    print ""
    print "Getting human similarities... Done."

    qToExecute = list()  # Reset tasks
    print "Getting chicken similarities..."
    for i in range(l1,l):
        print i," ",
        L[i] = 1 #0 for chickens
        qToExecute.append((i,l,S,ngg))

    # pThreadPool.map(__getSimilaritiesForIndex, qToExecute)
    map(__getSimilaritiesForIndex, qToExecute)

    # for i in range(l1,l):
    #     print i," ",
    #     L[i] = 1 #1 for chickens
    #     for j in range(i,l):
    #         S[i,j] = sop.getSimilarityDouble(ngg[i],ngg[j])
    print ""
    print "Getting chicken similarities... Done"

    # Update symmetric matrix, based on current findings
    for i in range(0,l):
        for j in range(0,i):
            S[i,j] = S[j,i]

    print "Similarity matrix constructed.."
    
    if not os.path.exists('SimilaritiesAndDictionaries'):
        os.mkdir('SimilaritiesAndDictionaries')

    np.savez('SimilaritiesAndDictionaries/UCNE.npz', hd=hd, cd=cd, l1=l1, l2=l2,
 l=l, S=S)


reps = 10

L1 = L[0:l1]
L2 = L[l1:]

metrics = dict()
cm = dict()


class_types = {0:"No kernelization",1:"Spectrum Clip",2:"Spectrum Flip",3:"Spectrum Shift",4:"Spectrum Square"}
print "Testing for different kernelization methods..\n\n"
for i in range(0, len(class_types)):
    try:
        print class_types[i],"\n"
        evaluator = cl.Evaluator(cl.SVM())
        Sp = cl.kernelization(S,i)
        S1 = Sp[0:l1,:]
        S2 = Sp[l1:,:]
        metrics[class_types[i]],cm[class_types[i]] = evaluator.Randomized_kfold((S1,S2),(L1,L2),reps,verbose=True)
        print ""
    except Exception as e:
        print "Approach %s failed for reason:\n%s"%(class_types[i], str(e))

np.savez('SimilaritiesAndDictionaries/metrics.npz', metrics=metrics, cm=cm)
