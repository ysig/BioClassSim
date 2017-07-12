import Bio.PDB as PDB
import numpy as np
import sys
sys.path.append('../../')
from source import bioRead as br
from source import proximityGraph as pg

import matplotlib.pyplot


reader = br.PDBreader()

reader.read('5mss.pdb')

q = reader.getDictionary()

for p in q:
    # what is an interesting distance between two aminoacids
    a = pg.ProximityGraph(q[p],3,2)
    a.GraphDraw()

