import os
import sys
import networkx as nx
sys.path.append(os.path.join(os.path.dirname(__file__),'../../..'))
from PyINSECT import representations as REP
import scipy.spatial as spatial


def decompose_tl(l):
    if(l==[]):
        return [],[]
    else:
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
        Data, Metric = decompose_tl(list(Data_Metric))
        self._Dwin = abs(int(Dwin))
        self._n = abs(int(n))
        self.setData(Data)
        self.setm(Metric)
        self._GPrintVerbose = GPrintVerbose
        if(not (self._Data == [])):
            _maxW = 0
            _minW = float("inf")
            self.buildGraph()

    #sets metric characteristics
    def setm(self,m):
        if not(m == []):
            self._m = list(m)
            self._ms = len(self._m)

    # creates the spatial KD tree needed
    def makeSTree(self):
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
        self.makeSTree()
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

    # interface wise function that produces the point needed to getIndeces
    def getPoint(self,i):
        return self._m[i]

    # interface wise function that produces indeces needed to findWindow
    def getIndeces(self,i):
        return self._point_tree.query_ball_point(self.getPoint(i), self._Dwin)

    # windowing is done based on first to last 
    # find's a window corresponding the current location (based on index)
    def findWindow(self,i):
        # based on index find all indexes in a distance of Dwin
        indeces = self.getIndeces(i)
        ngd = self._ngd
        win = list()
        # print "For i = "+str(i)+", indeces are:\n",indeces
        for i in indeces:
            # if for that index there exists an ngram
            # (for that check ngd)
            # concat it to the window
            key = self.makeKey(i)
            if(ngd.has_key(key)):
                win += ngd[key]
        # print "For i = "+str(i)+", window is :\n",win
        return win

    def makeKey(self,i):
        return self._Data[i]

    # ngd: stores based from ngram finals
    # lists of ngrams 
    # if ngram final exists inside ngd
    # append it to ngds end
    # else create a list
    # with only this element on the
    # corresponding ngds position
    def ngd_add(self,*args):
        l = args[0]
        if(self._ngd.has_key(l[-1])):
            self._ngd[l[-1]].append([str(l[:])])
        else:
            self._ngd[l[-1]] = [str(l[:])]

    # creates ngram's of window based on @param n
    # for building ngrams we consider the following:
    # because ngram step is always 1
    # so we always take an letter and pop
    # one out if there exists an ngram starting with
    # the symbol on position "index" of data
    # the corresponding ngram starting with
    # that index will be on position "index" also.
    def build_ngram(self,d = []):
        self._ngd = dict()
        self.setData(d)
        Data = self._Data    
        bound = min(self._n,self._dSize)
        # bound is the fixed ngram distance
        self._bound = bound
        l = Data[0:bound]
        # first ngd is the first element
        self.ngd_add(l,0)
        q = []
        q.append(l[:])
        if(self._n<self._dSize):
            i = 1
            for d in Data[min(self._n,self._dSize):]:
                l.pop(0)
                l.append(d)
                self.ngd_add(l,i)
                q.append(l[:])
                i+=1
        self._ngram = q
        return q
        
class CenterGraph(ProximityGraph,object):
    # add this method on proximity graph for setting m    
    def middle_flag(self,start):
        win = self._bound
        if (win%2==0):
            q = win/2
            # add to mkd Tree plus to indexes
            self._m+=np.divide(np.add(self._m[q+start],self._m[q+1+start]),2)
            self._ms+=1
            index = self._dSize
        else:
            index = (win-1)/2 +start
            #return index and position of the middle
        return index
    
    # builds Point out of the ngram middle
    def getPoint(self,i):
        if(self._bound%2==0):
            return np.divide(np.add(self._m[i+self_bound],self._m[i+self_bound+1]),2)
        else:
            return self._m[(self._bound-1)/2 + i]

    # adds an element on ngd
    def ngd_add(self,*args):
        l = args[0] # list
        start = args[1] # start index
        self._ngd[self.middle_flag(start)] = [str(l[:])]

    def makeKey(self,i):
        return i

