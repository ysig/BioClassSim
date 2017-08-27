import Bio.PDB as PDB
import numpy as np
import sys
import os
sys.path.append('../../')
from source import bioRead as br
from source import proximityGraph as pg
import matplotlib.pyplot

if not os.path.exists('./CATH95.pdb.tar.gz'):
    os.system("wget http://pongor.itk.ppke.hu/benchmark/partials/repository/CATH95/CATH95.pdb.tar.gz")
os.system("tar -xf CATH95.pdb.tar.gz")
reader = br.PDBreader()

reader.read_folder(os.path.abspath('CATH95'))

q = reader.getDictionary()
print q.keys()

