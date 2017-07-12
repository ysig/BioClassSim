from Bio import SeqIO
import Bio.PDB as PDB
import numpy as np

class SequenceReader:
    
    def __init__(self):
        self._sequence_dictionary={}
    
    def read(self,input_file):
        fasta_sequences = SeqIO.parse(open(input_file),'fasta')
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            self._sequence_dictionary[name] = sequence
                
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
    def read(self,input_file):
        parser = PDB.PDBParser()
        structure = parser.get_structure('', input_file)
        ppb = PDB.PPBuilder()
        Model = dict()
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

    def getDictionary(self):
        return self._sequence_dictionary
 
