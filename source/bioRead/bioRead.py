from os import listdir
from os.path import isfile, join
from Bio import SeqIO
import Bio.PDB as PDB
import numpy as np

# general bioReader
class bioReader(object):
    # initialize sequence dictionary
    def __init__(self):
        self._sequence_dictionary={}
    # return sequence dictionary
    def getDictionary(self):
        return self._sequence_dictionary
    # reads input and stores it into dictionary
    def read(self,*vargs):
        pass

class SequenceReader(bioReader):
    # reads a single fasta sequence
    # information is stored in field 
    # called sequence dictionary
    # with keys the sequence labels
    # trim is an optional flag for
    # common label of all sequences
    def read(self,input_file,trim=None):
        # load file to a parser
        fasta_sequences = SeqIO.parse(open(input_file),'fasta')
        if(trim==None):
            # if there is nothing to trim from 
            # parse sequences as they are
            # by storing name and sequence as
            # string
            for fasta in fasta_sequences:
                name, sequence = fasta.id, str(fasta.seq)
                self._sequence_dictionary[name] = sequence
        else:
            # if there is sothing to trim from 
            # parse sequences as they are but when holding sequence name
            # as dictionary key trim as suggested
            for fasta in fasta_sequences:
                name, sequence = fasta.id, str(fasta.seq)
                self._sequence_dictionary[name.strip(str(trim))] = sequence
            
    # method that returns the sequence dictionary    
class PDBreader(bioReader):
    
    # returns a dictionary of protein names
    # where for each a list is assigned with tuples
    # of the specific peptide and its center in xyz coordinates	
    # center is calculated as the mean of all xyz coordinates
    # from atoms that compose the specific aminoacid
    # reads a single pdb file
    def read(self,input_file,clean=True, i=-1):
        # clean flag determines if
        if (clean):
            self._sequence_dictionary={}
        # initialise a parser
        parser = PDB.PDBParser()
        # give the pdb name start as a name
        structure = parser.get_structure(input_file.split(".")[0], input_file)
        # a builder for peptide chain
        ppb = PDB.PPBuilder()
        # Model dictionary (sometimes models have more than one structure)
        Model = dict()
        j =0
        for model in structure:
            for chain in model:
                mod = list()
                peptides = ppb.build_peptides(chain, aa_only=False)
                # for the specific chain build & get peptides
                # because aa_only is True peptide chain is 
                # returned as one
                pp = peptides[0]
                # get sequence (similar to peptides)
                seq = pp.get_sequence()
                i = 0
                for residue in pp:
                    # for each member of the peptide chain
                    f = False;
                    for atom in residue:
                        # calculate the mean position of all his atoms
                        if(f==False):
                            f=True
                            pos = np.array([atom.get_coord()])
                        else:
                            pos = np.append(pos,np.array([atom.get_coord()]),axis=0)
                    # keep the peptide name and mean-address as touple
                    mod.append((seq[i],np.mean(pos,axis=0)))
                    i+=1
            Model[str(model)] = mod
        self._sequence_dictionary = Model

    # reads a directory of pdb-files
    def read_folder(self,folder):
        # load all files
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        # empty the sequence dictionary
        self._sequence_dictionary={}
        d = dict()
        for f in files:
            # read a single file
            self.read(join(folder,f),clean=False)
            # keep only the first name 
            d[f.split(".")[0]] = self.getDictionary()['<Model id=0>']
        self._sequence_dictionary = d
