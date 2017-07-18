import numpy as np
import os
import sys
#import class files
sys.path.append('../../../')
from source import bioRead as br
from source import classify as cl
from source import graph as GR
#import PyInsect for measuring similarity
sys.path.append('../../../../')
from PyINSECT import comparators as CMP

# A python script to examine classification problem for 
# dataset PCB00033 on: http://pongor.itk.ppke.hu/benchmark/#/Browse

if os.path.exists('SimilaritiesAndDictionaries/PCB00033_pdb.npz'):
    npz = np.load('SimilaritiesAndDictionaries/PCB00033_pdb.npz')
    q = npz['q']
    s = npz['s']
    l = npz['l']
    indexes = npz['indexes'].item()
else:

    if not os.path.exists("CATH95.pdb.tar.gz"):
        os.system("wget http://pongor.itk.ppke.hu/benchmark/partials/repository/CATH95/CATH95.pdb.tar.gz")
    if not os.path.exists("CATH95"):
        os.system("tar -xf CATH95.pdb.tar.gz")
    if not os.path.exists("CATH95_C_A_kfold_14_0.3_filt_33.cast"):
        os.system("wget http://pongor.itk.ppke.hu/benchmark/partials/repository/CATH95/CATH95_C_A_kfold_14_0.3_filt_33.cast")
    
    if not os.path.exists("SimilaritiesAndDictionaries/PCB00033_pdb_dict.npz"):
        if not os.path.exists('SimilaritiesAndDictionaries'):
            os.mkdir('SimilaritiesAndDictionaries')

        # get dictionary
        reader = br.PDBreader()
        reader.read_folder(os.path.abspath("CATH95"))
        hd = reader.getDictionary()
        np.savez('SimilaritiesAndDictionaries/PCB00033_pdb_dict.npz', hd=hd)
    else:
        npz = np.load('SimilaritiesAndDictionaries/PCB00033_pdb_dict.npz')
        hd = npz['hd'].item()
    print "Dictionaries Gained"
        

    # read ascii table contaning labels for each experiment
    q = np.genfromtxt("CATH95_C_A_kfold_14_0.3_filt_33.cast", names=True, delimiter='\t', dtype=None)
    print "Cast Matrix read"
    
    #create ngram's (normal)
    ngg = dict()
    for (key,val) in hd.iteritems():
        ngg[key] = GR.ProximityGraph(3,5,val)
    print "Ngrams Constructed"
    #calculate similarities (as dictionary)
    sop = CMP.SimilarityNVS()
    
	dkeys = hd.keys()
	del hd
    l = len(dkeys)
    s = np.empty([l,l])
    indexes = dict()
    i=0
    for k in dkeys:
        indexes[k] = i
        j = 0
        for l in dkeys:
            if(indexes.has_key(l)):
                s[j,i] = s[i,j]
            else:
                s[i,j] = sop.getSimilarityDouble(ngg[k],ngg[l])
            j+=1
        i+=1
    print "Similarity Matrix created"
    del ngg
    np.savez('SimilaritiesAndDictionaries/PCB00033_pdb.npz', l=l, s=s, q=q, indexes=indexes)

# make label sets
num_of_experiments = len(list(q[0]))-1
experiments = dict()
for i in range(0,num_of_experiments):
    experiments[i] = dict()
    experiments[i]['train+'] = list()
    experiments[i]['train-'] = list()
    experiments[i]['test+'] = list()
    experiments[i]['test-'] = list()

for lin in q:
    line = list(lin)
    name = str(line[0])
    i=0
    for tag in line[1:]:
        if(tag==1):
            # positive train
            experiments[i]['train+'].append(indexes[name])
        elif(tag==2):
            # negative train
            experiments[i]['train-'].append(indexes[name])
        elif(tag==3):
            # positive test
            experiments[i]['test+'].append(indexes[name])
        elif(tag==4):
            # negative test
            experiments[i]['test-'].append(indexes[name])
        i+=1

evaluator = cl.Evaluator(cl.SVM())
# kernelize S if needed
for i in range(0,num_of_experiments):
    experiments[i]['ltr+'] = len(experiments[i]['train+'])
    experiments[i]['ltr-'] = len(experiments[i]['train-'])
    experiments[i]['lte+'] = len(experiments[i]['test+'])
    experiments[i]['lte-'] = len(experiments[i]['test-'])
    experiments[i]['train'] = experiments[i]['train+'] + experiments[i]['train-']
    tS = s[:,experiments[i]['train']]
    experiments[i]['Strain+'] = tS[experiments[i]['train+'],:]
    experiments[i]['Strain-'] = tS[experiments[i]['train-'],:]
    experiments[i]['Ltrain+'] = np.ones(experiments[i]['ltr+'])
    experiments[i]['Ltrain-'] = np.zeros(experiments[i]['ltr-'])
    experiments[i]['Stest+'] = tS[experiments[i]['test+'],:]
    experiments[i]['Stest-'] = tS[experiments[i]['test-'],:]
    experiments[i]['Ltest+'] = np.ones(experiments[i]['lte+'])
    experiments[i]['Ltest-'] = np.zeros(experiments[i]['lte-'])        
        
    print str(i+1)+"th experiment with:\n#train+ = "+str(experiments[i]['ltr+'])+", #train- = "+str(experiments[i]['ltr-'])+"\n#test+ = "+str(experiments[i]['lte+'])+", #test- = "+str(experiments[i]['lte-'])
    print "\nDisplaying metrics"
    experiments[i]['metrics'],experiments[i]['confusion_matrix'] = evaluator.single(np.append(experiments[i]['Strain-'], experiments[i]['Strain+'] ,axis=0),np.append(experiments[i]['Ltrain-'],experiments[i]['Ltrain+'],axis=0),np.append(experiments[i]['Stest-'],experiments[i]['Stest+'],axis=0),np.append(experiments[i]['Ltest-'],experiments[i]['Ltest+'],axis=0),has_dummy=True)
    cl.displayMetrics(experiments[i]['metrics'],0)
    print "\n"
np.savez('SimilaritiesAndDictionaries/experiments_pdb.npz', experiments=experiments)
