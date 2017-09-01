import Bio.PDB as PDB
import numpy as np
import sys
sys.path.append('../../')
from source import bioRead as br
from source import proximityGraph as pg

import matplotlib.pyplot


reader = br.PDBreader()

reader.read('1l2y.pdb')

q = reader.getDictionary()

# what is an interesting distance between two aminoacids?
Dwin = float(raw_input('Give window distance: '))

# proximityy graph
a = pg.ProximityGraph(3,Dwin,q['<Model id=0>'])
a.GraphDraw()

# center graph
b = pg.CenterGraph(3,Dwin,q['<Model id=0>'])
b.GraphDraw()

# mean center graph
c = pg.MeanCenterGraph(3,Dwin,q['<Model id=0>'])
c.GraphDraw()
