import Bio.PDB as PDB
import numpy as np
import sys
import os
sys.path.append('../../')
from source import bioRead as br
from source import proximityGraph as pg
import matplotlib.pyplot

# a test to check if fasta sequences produced
# from pdb are the same with the ones from 
# the original fasta file

def keep2nd(d):
    q = dict()
    for (key,val) in d.iteritems():
        x = zip(*val)
        q[key] = ''.join(list(x[0]))
    return q

if not os.path.exists('CATH95.fasta'):
    os.system('wget http://pongor.itk.ppke.hu/benchmark/partials/repository/CATH95/CATH95.fasta')

if not os.path.exists("PCB00033_pdb_dict.npz"):
        # get dictionary
        if not os.path.exists("CATH95.pdb.tar.gz"):
            os.system("wget http://pongor.itk.ppke.hu/benchmark/partials/  repository/CATH95/CATH95.pdb.tar.gz")
        if not os.path.exists("CATH95"):
            os.system("tar -xf CATH95.pdb.tar.gz")
        reader = br.PDBreader()
        reader.read_folder(os.path.abspath("CATH95"))
        hd = reader.getDictionary()
        np.savez('PCB00033_pdb_dict.npz', hd=hd)
    else:
        npz = np.load('PCB00033_pdb_dict.npz')
        hd = npz['hd'].item()
    print "Dictionaries Gained"

snd = keep2nd(hd)
print "Unziped"

sr = br.SequenceReader()
sr.read('CATH95.fasta',trim = 'pdb|')
d = sr.getDictionary()
print "Fasta dictionary loaded"
i = 0
acc = 0
for (key,val) in snd.iteritems():
    i+=1
    if(d[key] != val):
        print "key: "+str(key)+" error!"
    else:
        acc += 1
        print "key: "+str(key)+" ok."
        
print "Accuracy is "+str(((acc*1.0)/i)*100)+"%"

