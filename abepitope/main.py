### IMPORTS ###
import math
import subprocess
import numpy as np
import torch
import torch.nn as nn
#import esm
import csv
from pathlib import Path
import sys
import pdb
import pickle
from pathlib import Path
import sys
MODULE_DIR = str( Path( Path(__file__).parent.resolve() ) )
sys.path.append(MODULE_DIR)
from esmif1_utilities import ESMIF1Model
from biopdb_utilities import identify_abag_with_hmm, get_abag_interaction_data
from utilities import load_pickle_file

### STATIC PATHS ###
ROOT_DIRECTORY = Path( Path(__file__).parent.resolve() )
MODELS_DIRECTORY = ROOT_DIRECTORY / "models"
ABEPISCORE_MODELS = MODELS_DIRECTORY / "abepiscore1.0"
ABEPITARGET_MODELS = MODELS_DIRECTORY / "abepitarget1.0"
AB_IDENTIFY_HMM_MODELS = MODELS_DIRECTORY / "hmm_antibody_identification"

### SET GPU OR CPU ###
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU device detected: {device}")
else:
    device = torch.device("cpu")
    print(f"GPU device not detected. Using CPU: {device}")

### MODEL ###


class DenseNet(nn.Module):
    """
    DenseNet used for finetuning ESM-IF1 to AbEpiScore-1.0 and AbEpiTarget-1.0
    Expect input of dimension: (batch_size, embedding size)
    """
    def __init__(self,
                 embedding_size = 512,
                 fc1_size = 300,
                 fc2_size = 150,
                 fc3_size = 100,
                 fc1_dropout = 0.6,
                 fc2_dropout = 0.65,
                 fc3_dropout = 0.5,
                 num_classes = 1):

        super(DenseNet, self).__init__()
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.fc1_dropout = fc1_dropout
        self.fc2_dropout = fc2_dropout
        self.fc3_dropout = fc3_dropout
        ff_input_size = embedding_size
      

        self.ff_model = nn.Sequential(nn.Linear(ff_input_size, fc1_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc1_dropout),
                                      nn.Linear(fc1_size, fc2_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc2_dropout),
                                      nn.Linear(fc2_size, fc3_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc3_dropout),
                                      nn.Linear(fc3_size, num_classes))


    def forward(self, X):
        """
        X: list of tensors. Each tensor, (embed size)
        """

        x = torch.stack(X)
        x = x.to(device)
        output = self.ff_model(x)

        return output


### CLASSES ###

class StructureData():
    def __init__(self):
        """
   
        """
    
    def encode_proteins(self, structure_directory, enc_directory, tmp_directory, atom_radius=4, esmif1_modelpath=None):
        # a single pdb file
        if structure_directory.is_file(): structure_files = [structure_directory]

        #directory with pdb files
        elif structure_directory.is_dir():
            structure_files = list(structure_directory.glob("**/*.pdb")) + list(structure_directory.glob("**/*.PDB")) + list(structure_directory.glob("**/*.cif")) + list(structure_directory.glob("**/*.CIF"))

        else:
            print(f"Structure(s) input did not exist. Not a file or a directory. Exiting!")
            return 
         
        esmif1_enc_files, esmif1_interface_encs, abag_sequence_data = [], [], []
        epitope_datas, paratope_datas = [], []
        success_files, failed_files = [], [] 

        if not enc_directory.is_dir(): enc_directory.mkdir(parents=True)
        if not tmp_directory.is_dir(): tmp_directory.mkdir(parents=True)

        #this will load the esmif1 model + alphabet
        print("Loading ESM-IF1 model...")
        esmif1_util = ESMIF1Model(esmif1_modelpath=esmif1_modelpath)
        print("Loading ESM-IF1 model... DONE")
    
        for structure_file in structure_files:

            #compute esmif1 encoding for protein
            print(f"Encoding structure from file: {structure_file.name}...")
            esmif1_enc_file = enc_directory / f"{structure_file.stem}.pickle"
            esmif1_enc_data = esmif1_util.compute_esmif1_on_protein(structure_file)
            
            with open(esmif1_enc_file, "wb") as outfile: pickle.dump(esmif1_enc_data, outfile, protocol=-1) 
            esmif1_enc_data = load_pickle_file(esmif1_enc_file)
            print(f"Encoding structure from file: {structure_file.name}... DONE")

            esmif1_enc = esmif1_enc_data["encs"]
            esmif1_seqs = esmif1_enc_data["seqs"]
            if esmif1_enc == None:
                print(f"Structure file {structure_file.name} was not detected as pdb or cif file. No score will be computed for it.")
                failed_files.append((structure_file.name, "Not a PDB/CIF"))
                continue

            nr_seqs = len(esmif1_enc)
            enc_idx_lookup = {esmif1_enc_data["chain_ids"][i]:i for i in range(nr_seqs)}
            
            print("Extracting interface antibody-antigen inteface encoding... ")
            #identify heavy, light and antigen chains 
            heavy_chain, light_chain, antigen_chains = identify_abag_with_hmm(structure_file, AB_IDENTIFY_HMM_MODELS, tmp_directory, pdb_id=structure_file.stem, verbose=False)
            antibody_chains = heavy_chain + light_chain

            if not antibody_chains and not antigen_chains:
                print(f"For structure file {structure_file.name} a light and heavy chain or a single-chain variable fragment (scFv), along with one or more antigen chains was not found.")
                failed_files.append((structure_file.name, "Light/heavy chain or antigen chain not detected"))
                continue
            #extract interface encoding
            epitope_data, paratope_data = get_abag_interaction_data(antigen_chains, antibody_chains, return_bio_pdb_aas=False, atom_radius=atom_radius)
            if not antibody_chains and not antigen_chains:
                print(f"For structure file {structure_file.name} no interface was detected {atom_radius}")
                failed_files.append((structure_file.name, "AbAb Interface not detected {atom_radius} Å"))
                continue

            # extra check (shouldn't happen)
            epipara_data = epitope_data + paratope_data
            check = all([True if esmif1_seqs[enc_idx_lookup[d[1]]][d[2]] == d[0] else False for d in epipara_data ])
            if not check: 
                print("ESMIF1 encoding and PDB read mismatch")


            epitope_enc = [esmif1_enc[enc_idx_lookup[e[1]]][e[2]] for e in epitope_data]
            paratope_enc = [esmif1_enc[enc_idx_lookup[p[1]]][p[2]] for p in paratope_data]
            interface_enc = torch.stack(epitope_enc + paratope_enc)
            interface_enc_avg = torch.mean(interface_enc, axis=0)

            epitope_datas.append(epitope_data)
            paratope_datas.append(paratope_data)
        
            esmif1_enc_files.append(esmif1_enc_file)
            esmif1_interface_encs.append(interface_enc_avg)
            abag_sequence_data.append( (f"{structure_file.name}_{''.join(esmif1_enc_data['chain_ids'])}", ":".join(esmif1_seqs) ) )
            success_files.append(structure_file)
            print("Extracting interface antibody-antigen inteface encoding... DONE")

        self.structure_files = success_files
        self.failed_files = failed_files
        self.esmif1_enc_files = esmif1_enc_files
        self.esmif1_interface_encs = esmif1_interface_encs
        self.epitope_datas = epitope_datas
        self.paratope_datas = paratope_datas
        self.abag_sequence_data = abag_sequence_data
            

class EvalAbAgs():

    def __init__(self,
                 structuredata,
                 device = None, decimal_precision=None):
        """
        Inputs and initialization:
            structuredata: Structure class object
            device: pytorch device to use, default is cuda if available else cpu.
        """
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.structuredata = structuredata
        self.decimal_precision= decimal_precision

    def predict(self, outpath):

        abepiscore_scores, _ = self.abepiscore()
        abepitarget_scores, _ = self.abepitarget()

        print("Creating output files...")
        structure_files = self.structuredata.structure_files
        epitope_datas = self.structuredata.epitope_datas
        paratope_datas = self.structuredata.paratope_datas
        
        # write main output files (.csv with scores and .csv abag interface annotation)
        self.create_csvfile(structure_files, abepiscore_scores, abepitarget_scores, epitope_datas, paratope_datas, outpath)
        
        # write sequence data as fasta file
        abag_sequence_data  = self.structuredata.abag_sequence_data
        self.write_sequence_data(abag_sequence_data, outpath / "abag_sequence_data.fasta")

        # write fail files
        faildata = self.structuredata.failed_files
        if faildata: self.write_failed_files(faildata, outpath / "failed_files.csv")
            
        
        print("Creating output files... DONE")

        
    def abepiscore(self):
        """
      
        """
        
        print("Running AbEpiScore-1.0")
        model = DenseNet().to(device)
        modelstates = list(ABEPISCORE_MODELS.glob("*"))
        nr_models = len(modelstates)
        interface_encs = self.structuredata.esmif1_interface_encs
        structure_files = self.structuredata.structure_files
        
        nr_structures = len(interface_encs)
        
        model_outputs = []
        model_outputs = torch.zeros((nr_models, nr_structures))
        for i in range(nr_models):
            model_state = modelstates[i]
            with torch.no_grad():
                model.load_state_dict(torch.load(model_state, map_location=self.device))
                model = model.to(self.device)
                model.eval()
                model_output = model(interface_encs)
                model_outputs[i] = torch.flatten(model_output).detach().cpu()

        avg_model_outputs = torch.mean(model_outputs, axis=0)
        print("Running AbEpiScore-1.0... DONE")
        
        return avg_model_outputs, structure_files


    def abepitarget(self):
        """
      
        """

        print("Running AbEpiTarget-1.0...")
        model = DenseNet(fc1_size=450, fc2_size=250, fc3_size=50, num_classes=2).to(device)
        modelstates = list(ABEPITARGET_MODELS.glob("*"))
        softmax_function = nn.Softmax(dim=1).to(device)          

        nr_models = len(modelstates)
        interface_encs = self.structuredata.esmif1_interface_encs
        structure_files = self.structuredata.structure_files
        nr_structures = len(interface_encs)
        
        model_outputs = []
        model_outputs = torch.zeros((nr_models, nr_structures))
        for i in range(nr_models):
            model_state = modelstates[i]
            with torch.no_grad():
                model.load_state_dict(torch.load(model_state, map_location=self.device))
                model = model.to(self.device)
                model.eval()    
                model_output = model(interface_encs)
                model_probs = softmax_function(model_output)[:, 1].detach().cpu()
                model_outputs[i] = model_probs 
                
                
        avg_model_outputs = torch.mean(model_outputs, axis=0)
        print("Running AbEpiTarget-1.0... DONE")

        return avg_model_outputs, structure_files

    def create_csvfile(self, structure_files, abepiscore_scores, abepitarget_scores, epitope_datas, paratope_datas, outpath):

        #create .csv content
        scores = list( zip(structure_files, abepiscore_scores, abepitarget_scores)) 
        interfaces = list( zip(structure_files, epitope_datas, paratope_datas) )
        score_csv_content, interface_csv_content = ["FileName,AbEpiScore-1.0,AbEpiTarget-1.0"], ["FileName,EpitopeResidue,ParatopeResidue"]

        for score in scores:
            structure_file, abepiscore_score, abepitarget_score = score

            if self.decimal_precision != None: abepiscore_score, abepitarget_score = str(round(abepiscore_score.item(), self.decimal_precision)), str(round(abepitarget_score.item(), self.decimal_precision) )
            else: abepiscore_score, abepitarget_score = str(abepiscore_score.item()), str(abepitarget_score.item())
            
            score_csv_content.append(f"{structure_file.name},{abepiscore_score},{abepitarget_score}")


        for interface in interfaces:
            structure_file, epitope_data, paratope_data = interface
            nr_residues = len(epitope_data)
            interface_csv_content.extend( [f"{structure_file.name},{epitope_data[i]},{paratope_data[i]}" for i in range(nr_residues)])
        

        score_csv_content = "\n".join(score_csv_content)
        interface_csv_content = "\n".join(interface_csv_content)

        #write to .csv file 
        if not outpath.is_dir(): outpath.mkdir(parents=True)
        with open(outpath / "output.csv", "w") as outfile: outfile.write(score_csv_content)
        with open(outpath / "interface.csv", "w") as outfile: outfile.write(interface_csv_content)
    
    def write_failed_files(self, faildata, out_filepath):
        failfile_content = "FileName,Reason\n"
        failfile_content += "\n".join([f"{d[0]},{d[1]}" for d in faildata])
        with open(out_filepath, "w") as outfile: outfile.write(failfile_content) 

    def write_sequence_data(self, sequence_data, out_filepath):
        N = len(sequence_data)
        fastafile_content = "\n".join([f">{sequence_data[i][0]}\n{sequence_data[i][1]}" for i in range(N)])
        with open(out_filepath, "w") as outfile: outfile.write(fastafile_content)