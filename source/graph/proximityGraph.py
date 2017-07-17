import os
import sys
import networkx as nx
sys.path.append(os.path.join(os.path.dirname(__file__),'../../..'))
from PyINSECT import representations as REP
import scipy.spatial as spatial

def decompose_tl(l):
    list1 = list()
    list2 = list()
    for (a,b) in l:
        list1.append(a)
        list2.append(b)
    return list1,list2

class ProximityGraph(REP.DocumentNGramGraph,object):
    def __init__(self, n=3, Dwin=2, Data_Metric = [], GPrintVerbose = True):
        # distance_metric is a list of touples
        # containing symbols and their 3D coordinates
        Data, self._m = decompose_tl(list(Data_Metric))
        if(self._m==[]):
            self._point_tree = None
        else:
            self._point_tree = spatial.cKDTree(self._m)
        super(self.__class__, self).__init__(n, Dwin, Data, GPrintVerbose)
    
    def setm(self,m):
        if not(m == []):
            self._m = list(m)
            self._point_tree = spatial.cKDTree(self._m)
            
    # we will now define @method buildGraph
    # which takes a data input
    # segments ngrams
    # and creates ngrams based on a given window
    # !notice: at this developmental stage the weighting method
    # may not be correct
    def buildGraph(self,verbose = False, d=[]):
        # set Data @class_var
        Data, m = decompose_tl(d)
        self.setm(m)
        self.setData(d)
        Data = self._Data
        
        # build ngram
        ng = self.build_ngram()
        s = len(ng)
    
        win = self._Dwin
        
        self._Graph = nx.DiGraph()
        self._edges = set()
        
        o = min(self._Dwin,s)
        if(o>=1):
            # need to change window associated with dm
            # append the first full window
            # while adding the needed edges
            i = 0
            for gram in ng:
                window = self.findWindow(i)
                for w in window:
                    self.addEdgeInc(gram,w)
                i+=1
            # print graph (optional)
            if verbose:
                self.GraphDraw(self._GPrintVerbose)
        return self._Graph

    # windowing is done based on first to last 
    # find's a window corresponding the current location (based on index)
    def findWindow(self,i):
        # based on index find all indexes in a distance of Dwin
        indeces = self._point_tree.query_ball_point(self._m[i], self._Dwin)
        ngf = self._ngf
        win = list()
        for i in indeces:
            # if for that index there exists an ngram
            # (for that check ngf)
            # concat it to the window 
            if(ngf.has_key(self._Data[i])):
                win += ngf[self._Data[i]]
        return win

    # creates ngram's of window based on @param n
    # for building ngrams we consider the following:
    # because ngram step is always 1
    # so we always take an letter and pop
    # one out if there exists an ngram starting with
    # the symbol on position "index" of data
    # the corresponding ngram starting with
    # that index will be on position "index" also.
    def build_ngram(self,d = []):
        # ngf: stores based from ngram finals
        # lists of ngrams 
        ngf = dict()
        self.setData(d)
        Data = self._Data
        l = Data[0:min(self._n,self._dSize)]
        # first ngf is the first element
        ngf[l[-1]] = [str(l[:])]
        q = []
        q.append(l[:])
        if(self._n<self._dSize):
            i = 1
            for d in Data[min(self._n,self._dSize):]:
                l.pop(0)
                l.append(d)
                # if ngram final exists inside ngf
                # append it to ngfs end
                # else create a list
                # with only this element on the
                # corresponding ngfs position
                if(ngf.has_key(l[-1])):
                    ngf[l[-1]].append([str(l[:])])
                else:
                    ngf[l[-1]] = [str(l[:])]
                q.append(l[:])
                i+=1
        self._ngram = q
        self._ngf = ngf
        return q
