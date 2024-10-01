from biopdb_utilities import read_pdb_structure, is_pdb_file_biopython, is_cif_file_biopython, read_cif_structure
import esm
import torch
from pathlib import Path
import sys
import time

MODULE_DIR = str( Path( Path(__file__).parent.resolve() ) )
sys.path.append(MODULE_DIR)

class ESMIF1Model():
    
    def __init__(self, device=None, esmif1_modelpath=None):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        model_name = "esm_if1_gvp4_t16_142M_UR50"
        torch_hubdir = torch.hub.get_dir()
        torch_hubfile = Path(torch_hubdir) / "checkpoints" / f"{model_name}.pt"

        #download model from torchhub, (esmif1 source code hardcoded to load onto cpu)
        if esmif1_modelpath is None and not torch_hubfile.is_file():
            esmif1_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()       

        #if model already in torchhub, load directly onto device for speed up 
        elif esmif1_modelpath is None and torch_hubfile.is_file():
            model_data = torch.load(str(torch_hubfile), map_location=self.device)
            esmif1_model, alphabet = esm.pretrained.load_model_and_alphabet_core(model_name, model_data, None)

        #load model from specified location
        else:
            model_data = torch.load(str(esmif1_modelpath), map_location=self.device)
            esmif1_model, alphabet = esm.pretrained.load_model_and_alphabet_core(model_name, model_data, None)

        self.esmif1_model = esmif1_model.eval().to(self.device)
        self.alphabet = alphabet

    def compute_esmif1_on_protein(self, pdb_file):
        """
        """

        esmif1_encs, sequence_order, chain_ids = self.compute_esmif1_on_pdb(pdb_file)        
        #save esm-if1 encoding
        esmif1_enc_data = {"encs": esmif1_encs, "chain_ids": chain_ids, "seqs": sequence_order}

        return esmif1_enc_data 


    def compute_esmif1_on_pdb(self, pdb_file):


        if is_pdb_file_biopython(pdb_file): structure = read_pdb_structure(pdb_file)
        elif is_cif_file_biopython(pdb_file): structure = read_cif_structure(pdb_file)
        else: structure = None

        if structure == None:
            return None, None, None
        
        esmif1_encs, sequence_order = [], []
        #load abag complex into esm-if1
        chain_names = [c.get_id() for c in structure.get_chains()]
        esmif1_loaded_structure = esm.inverse_folding.util.load_structure(str(pdb_file), chain_names)
        coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(esmif1_loaded_structure)
        
        #extract esmif1 encodings
        for c in chain_names:
            #get esmif1_encoding
            seq = native_seqs[c]    
            with torch.no_grad():
                esmif1_enc = self.esmif1encs_forwardpass(self.esmif1_model, self.alphabet, coords, c, seq) 

            esmif1_encs.append(esmif1_enc)
            sequence_order.append(seq)

        return esmif1_encs, sequence_order, chain_names



    def esmif1encs_forwardpass(self, model, alphabet, coords, c, seq):

        with torch.no_grad():
            #device = next(model.parameters()).device
            coord_cat = esm.inverse_folding.multichain_util._concatenate_coords(coords, c)
            batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
            
            batch = [(coord_cat, None, seq)]
            
            batch_coords, confidence, _, tokens, padding_mask = batch_converter(batch, device=self.device)
            prev_output_tokens = tokens[:, :-1].to(self.device) 

            # gvp transformer encoder forward pass 
            enc_out = model.encoder.forward(
                batch_coords,
                padding_mask,
                confidence, return_all_hiddens=False,)
            
            # prepare encoder outputs for saving
            target_chain_len = coords[c].shape[0]
            enc_out = enc_out["encoder_out"][0][1:-1, 0]
            enc_out = enc_out[:target_chain_len].cpu().detach()
            
            return enc_out


