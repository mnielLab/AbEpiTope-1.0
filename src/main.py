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
ANTIINTERNET_MODELS = MODELS_DIRECTORY / "antiinternet1.0"
ANTISCOUT_MODELS = MODELS_DIRECTORY / "antiscout1.0"
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
    DenseNet used for finetuning ESM-IF1 to AntiInterNet-1.0 and AntiScout-1.0
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
                 num_classes = 1, add_pTM=False, add_ipTM=False, add_hcdr3_I=False,
                 add_norm=False):
        super(DenseNet, self).__init__()


        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.fc1_dropout = fc1_dropout
        self.fc2_dropout = fc2_dropout
        self.fc3_dropout = fc3_dropout

        #increase FFNN inputs, if additional features are used
        ff_input_size = embedding_size
        if add_pTM: ff_input_size += 1
        if add_ipTM: ff_input_size += 1
        if add_hcdr3_I: ff_input_size += 1
        if ff_input_size != embedding_size: self.added_features = True
        else: self.added_features = False

        #normalize input features or not
        self.add_norm = add_norm

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
        #if additional features were added
        if self.added_features:
            x = [d[0] for d in X]
            add_features = [torch.tensor(d[1]) for d in X]

            #convert to batch dimension
            x = torch.stack(x)
            x = x.to(device)
            #create feature vector
            add_features = torch.stack(add_features) #.unsqueeze(dim=0)
            add_features = add_features.to(device)

            #apply z-score normalization
            if self.add_norm:
                add_features = (add_features - add_features.mean(dim=1, keepdim=True)) / add_features.std(dim=1, keepdim=True)
                x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
            #concatentate added features

            x = torch.cat((x, add_features), axis=1)

        #dont add additional featqures
        else:
            #convert to batch dimension
            x = torch.stack(X)
            x = x.to(device)
        output = self.ff_model(x)

        return output

### CLASSES ###

class StructureData():
    def __init__(self):
        """
        Initialize Antigens class object
        Inputs:
            device: pytorch device to use, default is cuda if available else cpu.
        """
    
    def encode_proteins(self, structure_directory, enc_directory, tmp_directory, atom_radius=4):
        #look for .pdb or .cif files
        if not structure_directory.is_dir(): 
            print(f"Specified structure directory: {structure_directory} did not exist.")
            return 

    
        pdb_files = list(structure_directory.glob("*.pdb"))
        cif_files = list(structure_directory.glob("*.cif"))
        structure_files = pdb_files + cif_files
        esmif1_enc_files, esmif1_interface_encs = [], []
        
        if not enc_directory.is_dir(): enc_directory.mkdir(parents=True)
        if not tmp_directory.is_dir(): tmp_directory.mkdir(parents=True)

        #this will load the esmif1 model + alphabet
        print("Loading ESM-IF1 model...")
        esmif1_util = ESMIF1Model()
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
            nr_seqs = len(esmif1_enc)
            enc_idx_lookup = {esmif1_enc_data["chain_ids"][i]:i for i in range(nr_seqs)}
            

            print("Extracting interface antibody-antigen inteface encoding... ")
            #identify heavy, light and antigen chains 
            heavy_chain, light_chain, antigen_chains = identify_abag_with_hmm(structure_file, AB_IDENTIFY_HMM_MODELS, tmp_directory, pdb_id=structure_file.stem, verbose=False)
            antibody_chains = heavy_chain + light_chain
            
            #extract interface encoding
            epitope_data, paratope_data = get_abag_interaction_data(antigen_chains, antibody_chains, return_bio_pdb_aas=False, atom_radius=atom_radius)
            epitope_enc = [esmif1_enc[enc_idx_lookup[e[1]]][e[2]] for e in epitope_data]
            paratope_enc = [esmif1_enc[enc_idx_lookup[p[1]]][p[2]] for p in paratope_data]
            interface_enc = torch.stack(epitope_enc + paratope_enc)
            interface_enc_avg = torch.mean(interface_enc, axis=0)
        
            esmif1_enc_files.append(esmif1_enc_file)
            esmif1_interface_encs.append(interface_enc_avg)
            print("Extracting interface antibody-antigen inteface encoding... DONE")

        self.structure_files = structure_files
        self.esmif1_enc_files = esmif1_enc_files
        self.esmif1_interface_encs = esmif1_interface_encs
            
        
class AntiInterNet():
    
    def __init__(self,
                 structuredata,
                 device = None):
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
        self.model = DenseNet()
        self.modelstates = list(ANTIINTERNET_MODELS.glob("*"))


    def ensemble_predict(self):
        """
        INPUTS: antigens: Antigens() class object.  
        
        OUTPUTS:
                No outputs. Stores probabilities of ensemble models in Antigens() class object.
                Run bp3_pred_variable_threshold() or bp3_pred_majority_vote() afterwards to make predictions. 
        """

        
        model = self.model
        modelstates = self.modelstates
        nr_models = len(modelstates)
        interface_encs = self.structuredata.esmif1_interface_encs
        structure_datafiles = self.structuredata.structure_files
        nr_structures = len(interface_encs)
        print("Running AntiInterNet")
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


        avg_model_output = torch.mean(model_outputs, axis=0)
        score_dict = {structure_datafiles[i].name:avg_model_output[i].item() for i in range(nr_structures)}
        
        return score_dict


class AntiScout():
    
    def __init__(self,
                 structuredata,
                 device = None):
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
        self.model = DenseNet(fc1_size=450, fc2_size=250, fc3_size=50, num_classes=2)
        self.modelstates = list(ANTISCOUT_MODELS.glob("*"))

    def ensemble_predict(self):
        """
        INPUTS: antigens: Antigens() class object.  
        
        OUTPUTS:
                No outputs. Stores probabilities of ensemble models in Antigens() class object.
                Run bp3_pred_variable_threshold() or bp3_pred_majority_vote() afterwards to make predictions. 
        """

        ensemble_probs = list()
        threshold_keys = list()
        softmax_function = nn.Softmax(dim=1).to(device)
        
        model = self.model
        modelstates = self.modelstates
        nr_models = len(modelstates)
        interface_encs = self.structuredata.esmif1_interface_encs
        structure_datafiles = self.structuredata.structure_files
        nr_structures = len(interface_encs)
        print("Generating AntiScout scores")
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
                
                
        avg_model_output = torch.mean(model_outputs, axis=0)
        score_dict = {structure_datafiles[i].name:avg_model_output[i].item() for i in range(nr_structures)}
        return score_dict 