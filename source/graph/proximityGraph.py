import os
import sys
import networkx as nx
sys.path.append(os.path.join(os.path.dirname(__file__),'../../..'))
from PyINSECT import representations as REP
import scipy.spatial as spatial
import numpy as np

# could be done by using zip founction
# in its inverse
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

# a general Proximity Graph interface
# plus an implemented (default)
# end to end greedy method
class ProximityGraph(REP.DocumentNGramGraph,object):
    def __init__(self, n=3, Dwin=2, Data_Metric = [], GPrintVerbose = True):
        # distance_metric is a list of touples
        # containing symbols and their 3D coordinates

        # Get Data and Metric
        Data, Metric = decompose_tl(list(Data_Metric))

        # Get Dwin float value.
        self._Dwin = float(Dwin)

        # get n as absolute value
        self._n = abs(int(n))

        # set Data variable with Data
        self.setData(Data)

        # set Metric variable with Metric
        self.setMetric(Metric)

        # initialise a tree metric list
        # (will be used for finding points)
        self._tree_metric = list()

        # type of graph printing when defined in the initialisation
        self._GPrintVerbose = GPrintVerbose

        # if Data is set: initialize maxW, minW and build.
        if(not (self._Data == [])):
            _maxW = 0
            _minW = float("inf")
            self.buildGraph()

    #sets metric characteristics
    def setMetric(self,Metric):
        if not(Metric == []):
            self._Metric = list(Metric)
            self._Metric_Size = len(self._Metric)

    # creates the spatial KD tree needed for point searching in 3D
    def makeSTree(self):
        self._point_tree = spatial.cKDTree(self._tree_metric)

    # we will now define @method buildGraph
    # which takes a data input
    # segments ngrams
    # and creates ngrams based on a given window
    # creates the spatial tree
    # and based on a window Dwin in 3D space
    # and a correspondence function finds
    # certain points.
    def buildGraph(self,verbose = False, d = []):
        # set Data @class_var
        Data, Metric = decompose_tl(d)

        # set Metric variable
        self.setMetric(Metric)
    
        # set Data variable
        self.setData(d)
        Data = self._Data
        
        # build ngram
        Ngrams = self.build_ngram()

        # creates the spatial Tree
        self.makeSTree()

        # size of ngrams
        ngram_size = len(Ngrams)
    
        # search window distance
        win = self._Dwin
        
        self._Graph = nx.DiGraph()
        self._edges = set()
        
        o = min(self._Dwin,ngram_size)
        if(o>=1):
            # need to change window associated with dm
            # append the first full window
            # while adding the needed edges
            i = 0
            for gram in Ngrams:
                # return based on index all ngrams that are in 
                # a window dwin from the current ngram ["gram"]
                window = self.findWindow(i)
                if str(gram) in window: window.remove(str(gram))
                for w in window:
                    # for every ngram inside the window
                    # add edge or if exists increment weight
                    self.addEdgeInc(str(gram),w)
                i+=1
            # print graph (optional)
            if verbose:
                self.GraphDraw(self._GPrintVerbose)
        return self._Graph

    # interface wise function that produces the point needed to getIndeces
    def getPoint(self,i):
        return self._Metric[i]

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

    # produce based on the ngrams index inside the spatial kdTree
    # the coresponding ngram dictionary key (which holds the ngram's names)
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
                # add to ngram dictionary
                # if tree metric is not dependent statically on metric
                # here is your chance, to define tree metric (piece by piece)
                self.ngd_add(l,i)
                q.append(l[:])
                i+=1

        # set tree metric (bypass function)
        self.set_tree_metric()
        self._ngram = q
        return q

    # a bypass function in when metric is all you need
    def set_tree_metric(self):
        self._tree_metric = self._Metric

# center graphs take into account only the central ngram
# for calculating in window ngrams
# If n is even two center elements are considered as one
class CenterGraph(ProximityGraph,object):

    # add this method on proximity graph for setting m
    def middle_flag(self,start):
        win = self._bound
        if (win%2==0):
            q = win/2

            # add to mkd Tree plus to indexes
            self._Metric+=np.divide(np.add(self._Metric[q+start],self._Metric[q+1+start]),2)
            self._Metric_Size+=1
            index = self._Metric_Size #?
        else:

            #return index and position of the middle
            index = (win-1)/2 +start

        return index
    
    # builds Point out of the ngram middle
    def getPoint(self,i):
        if(self._bound%2==0):
            return np.divide(np.add(self._Metric[i+self_bound],self._Metric[i+self_bound+1]),2)
        else:
            return self._Metric[(self._bound-1)/2 + i]

    # adds an element on ngd
    def ngd_add(self,*args):
        l = args[0] # list
        start = args[1] # start index
        self._ngd[self.middle_flag(start)] = [str(l[:])]

    # produce based on the ngrams index inside the spatial kdTree
    # the coresponding ngram dictionary key (which holds the ngram's names)
    def makeKey(self,i):
        return i

# Mean center graph considers calculates
# neighbor ngrams, by considering ngrams
# as objects in the mean 3D position
# of all the ngrams that constitute them
class MeanCenterGraph(ProximityGraph,object):

    # skip the bypass function
    def set_tree_metric(self):
        pass
    
    # builds Point out of the ngram middle
    def getPoint(self,i):
        return self._tree_metric[i]

    # adds an element on ngd
    def ngd_add(self,*args):
        l = args[0] # list
        start = args[1] # start index
        win = self._n
        Max_Data_Size = self._dSize
        # add to mkd Tree plus to indexes
        ngram_metrics = self._Metric[start:start+win]
        if(start+win >= Max_Data_Size or win==1):
           self._tree_metric += np.mean(ngram_metrics,axis=0)
        else:
           self._tree_metric += ngram_metrics
        self._ngd[start] = [str(l[:])]

    # produce based on the ngrams index inside the spatial kdTree
    # the corresponding ngram dictionary key (which holds the ngram's names)    
    def makeKey(self,i):
        return i

