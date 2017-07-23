import sys
#import class files
sys.path.append('../../')
from source import bioRead as br
from source import classify as cl
#import PyInsect for measuring similarity
sys.path.append('../../../')
from PyINSECT import representations as REP

sr = br.SequenceReader()
sr.read('Schizosaccharomyces_pombe.ASM294v2.30.dna.I.fa')
d = sr.getDictionary()

ngg1 = REP.DocumentNGramGraph(3,2,d["I"])
ngg1.GraphDraw(False) 
