from biopdb_utilities import read_pdb_structure
import esm
import torch
from pathlib import Path
import sys
MODULE_DIR = str( Path( Path(__file__).parent.resolve() ) )
sys.path.append(MODULE_DIR)

class ESMIF1Model():
    
    def __init__(self):

        esmif1_model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50() 
        self.esmif1_model = esmif1_model.eval()

    def compute_esmif1_on_protein(self, pdb_file):
        """
        """

        esmif1_encs, sequence_order, chain_ids = self.compute_esmif1_on_pdb(pdb_file)
        nr_chains = len(chain_ids)

        #temporaray placeholders
        sequence_order = [i for i in range(nr_chains)]
            
        #save esm-if1 encoding
        esmif1_enc_data = {"encs": esmif1_encs, "sequences":sequence_order, "chain_ids": chain_ids}
        #with open(esmif1_enc_out, "wbesmif1_enc_data) as outfile: pickle.dump(esmif1_enc_data, outfile, protocol=-1)

        return esmif1_enc_data 


    def compute_esmif1_on_pdb(self, pdb_file):

        structure = read_pdb_structure(pdb_file)
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
            device = next(model.parameters()).device
            coord_cat = esm.inverse_folding.multichain_util._concatenate_coords(coords, c)
            batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
            
            batch = [(coord_cat, None, seq)]
            
            batch_coords, confidence, _, tokens, padding_mask = batch_converter(batch, device=device)
            prev_output_tokens = tokens[:, :-1].to(device) 

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


