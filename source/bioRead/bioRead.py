from os import listdir
from os.path import isfile, join
from Bio import SeqIO
import Bio.PDB as PDB
import numpy as np

class SequenceReader:
    
    def __init__(self):
        self._sequence_dictionary={}
    
    def read(self,input_file,trim=None):
        fasta_sequences = SeqIO.parse(open(input_file),'fasta')
        if(trim==None):
            for fasta in fasta_sequences:
                name, sequence = fasta.id, str(fasta.seq)
                self._sequence_dictionary[name] = sequence
        else:
            for fasta in fasta_sequences:
                name, sequence = fasta.id, str(fasta.seq)
                self._sequence_dictionary[name.strip(str(trim))] = sequence
            
                
    def getDictionary(self):
        return self._sequence_dictionary
    
class PDBreader:

    def __init__(self):
        self._sequence_dictionary={}
    
    # returns a dictionary of protein names
    # where for each a list is assigned with tuples
    # of the specific peptide and its center in xyz coordinates	
    # center is calculated as the mean of all xyz coordinates
    # from atoms that compose the specific aminoacid
    def read(self,input_file,clean=True, i=-1):
        if (clean):
            self._sequence_dictionary={}
        parser = PDB.PDBParser()
        structure = parser.get_structure(input_file.split(".")[0], input_file)
        ppb = PDB.PPBuilder()
        Model = dict()
        j =0
        for model in structure:
            for chain in model:
                mod = list()
                peptides = ppb.build_peptides(chain, aa_only=False)
                pp = peptides[0]
                seq = pp.get_sequence()
                i = 0
                for residue in pp:
                    f = False;
                    for atom in residue:
                        if(f==False):
                            f=True
                            pos = np.array([atom.get_coord()])
                        else:
                            pos = np.append(pos,np.array([atom.get_coord()]),axis=0)
                    mod.append((seq[i],np.mean(pos,axis=0)))
                    i+=1
            Model[str(model)] = mod
        self._sequence_dictionary = Model

    def read_folder(self,folder):
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        d = dict()
        for f in files:
            self.read(join(folder,f),clean=False)
            d[f.split(".")[0]] = self.getDictionary()['<Model id=0>']
        self._sequence_dictionary = d
    def getDictionary(self):
        return self._sequence_dictionary
 
