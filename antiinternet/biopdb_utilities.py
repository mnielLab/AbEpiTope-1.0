from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch, Selection, Polypeptide
import subprocess
from pathlib import Path
import sys
import pdb
MODULE_DIR = str( Path( Path(__file__).parent.resolve() ) )
sys.path.append(MODULE_DIR)
AA3to1_DICT = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def identify_abag_with_hmm(abag_path, hmm_models_directory, tmp, pdb_id="foo", hmm_eval=float(1e-18), verbose=True):
    
    #make temporay result path, if it doesn't exist
    if not tmp.is_dir(): tmp.mkdir(parents=True)
    #set hidden markov models
    heavy_hmm = hmm_models_directory / "heavy.hmm"
    light_lambda_hmm = hmm_models_directory / "lambda.hmm"
    light_kappa_hmm = hmm_models_directory / "kappa.hmm"
    tmp_fastafile = tmp / "tmp.fasta"
    
    heavy_chain, light_chain, antigen_chains = [], [], [] 

    if is_pdb_file_biopython(abag_path): model = read_pdb_structure(abag_path)
    elif is_cif_file_biopython(abag_path): model = read_cif_structure(abag_path)

    else:
        print("Could not recognize {abag_path} as a pdb or cif file")
        return [], [], []


    for chain in model:

        #create a temporary fasta file for chain
        write_biopdb_chain_residues_to_fasta(chain, pdb_id, tgt_file = tmp / "chain_seq_tmp.fasta")
        #identify if it is a heavy / light chain or not.
        #do heavy and kappa HMM profile matching
        subprocess.run(["hmmsearch", "--noali", "-E", str(hmm_eval), "-o", tmp / "heavy_chains", heavy_hmm , tmp / "chain_seq_tmp.fasta"])
        subprocess.run(["hmmsearch", "--noali", "-E", str(hmm_eval), "-o", tmp / "lambda_chains", light_lambda_hmm, tmp / "chain_seq_tmp.fasta"])
        subprocess.run(["hmmsearch", "--noali", "-E", str(hmm_eval), "-o", tmp / "kappa_chains", light_kappa_hmm, tmp / "chain_seq_tmp.fasta"])

        heavy_check = ""
        lambda_check = ""
        kappa_check = ""

        try:
            heavy_check = subprocess.check_output(f"grep -oP '>> \K\w+' {str(tmp)}/heavy_chains", shell=True)
        except subprocess.CalledProcessError as error:
            pass

        try:
            lambda_check = subprocess.check_output(f"grep -oP '>> \K\w+' {str(tmp)}/lambda_chains", shell=True)
        except subprocess.CalledProcessError as error:
            pass

        try:
            kappa_check = subprocess.check_output(f"grep -oP '>> \K\w+' {str(tmp)}/kappa_chains", shell=True)
        except subprocess.CalledProcessError as error:
            pass

        chain_id = chain.get_id()
        #heavy chain
        if heavy_check and not lambda_check and not kappa_check:
            if verbose: print(f"Heavy chain: {chain_id}")
            heavy_chain.append(chain)

        #light chain
        elif (lambda_check or kappa_check) and not heavy_check:
            if verbose: print(f"Light chain: {chain_id}")
            light_chain.append(chain)
           
        
        #identified as both heavy and light
        elif heavy_check and (lambda_check or kappa_check):
            if verbose: print(f"Identified as both heavy and light chain: {chain_id}")
            heavy_chain.append(chain)
        
        elif not heavy_check and not lambda_check and not kappa_check:
            if verbose: print(f"Antigen chain: {chain_id}")
            antigen_chains.append(chain)


    return heavy_chain, light_chain, antigen_chains

def is_pdb_file_biopython(file_path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('pdb_structure', file_path)
        return True  # If no exception is raised, it's a valid PDB file
    except Exception as e:
        return False


def is_cif_file_biopython(file_path):
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure('pdb_structure', file_path)
        return True  # If no exception is raised, it's a valid PDB file
    except Exception as e:
        return False

def read_pdb_structure(pdb_file, pdb_id="foo", modelnr=0, return_all_models = False):
        """
        pdb_id: PDB acession, string
        pdb_file: path to 

        """
        #reading model 0 by default
    
        assert isinstance(modelnr, int), f"Model number needs to be a valid integer, it was {modelnr}"
        parser = PDBParser()
        structure = parser.get_structure(pdb_id, pdb_file)

        #return all models
        if return_all_models:
            models = list()
            for m in structure: models.append(m)
            return models

        #return only desired model
        else: return structure[modelnr]

def read_cif_structure(cif_file, pdb_id="foo", modelnr=0, return_all_models=False):
    """
    Reads a CIF file and returns a specific model or all models in the structure.
    
    cif_file: path to the CIF file.
    pdb_id: PDB accession, string (default is 'foo').
    modelnr: The model number to return (default is 0).
    return_all_models: If True, returns all models in the structure.
    
    Returns the specified model or all models if return_all_models is True.
    """
    assert isinstance(modelnr, int), f"Model number needs to be a valid integer, it was {modelnr}"
    parser = MMCIFParser()
    structure = parser.get_structure(pdb_id, cif_file)

    if return_all_models:
        models = list(structure.get_models())
        return models
    else:
        return structure[modelnr]

def write_biopdb_chain_residues_to_fasta(chains, pdb_acc_name, tgt_file=None):
    """
    Inputs: residues: List of Bio PDB residue class objects.
            epitope_or_paratope_res: List of Bio PDB residue class objects
    Outputs: annotated_AA_seq: String of amino acid characters where,
                               capitalized letter = Epitope/Paratope residue
                               non-capitalized letter = Non-epitope/Non-paratopee residue.
                               x/X are non amino acid characters, such as DNA or RNA.
    """
    #sometimes chains are passed as just one chain not in a list
    if not isinstance(chains, list): chains = [chains]
    
    fasta_file_content = str()
    AA_seqs = list()
    #remove hetero atoms
    for chain in chains: get_and_remove_heteroatoms(chain)

    for chain in chains:

        chain_id = chain.get_id()
        chain_residues = Selection.unfold_entities(chain, "R")
        
        AA_seq = str()

        for residue in chain_residues:
        
            try:
                aa = AA3to1_DICT[residue.get_resname()]
            #when residue is something nonstandard
            except KeyError:
                print(aa)
                print("Non-standard amino acid detected")
                aa = "X"
    
            AA_seq += aa

        #append to fasta_format
        fasta_file_content += f">{pdb_acc_name}_{chain_id}\n{AA_seq}\n"
        AA_seqs.append(AA_seq)
    if tgt_file != None:
        with open(tgt_file, "w") as outfile:
            outfile.write(fasta_file_content[:-1])

    return AA_seqs


def get_and_remove_heteroatoms(chain):
    """
   Heteroatoms in the form of water and other solvents need to be removed from the chain.
   Inputs: chain id

    """
    residues = Selection.unfold_entities(chain, "R")
    heteroatom_residue_ids = list()
    for residue in residues:
        residue_id = residue.get_full_id()

        #residue is a heteroatom
        if residue_id[3][0] != " ":
            heteroatom_residue_ids.append(residue_id[3])
    #remove all heteroatoms
    [chain.detach_child(ids) for ids in heteroatom_residue_ids]

def get_abag_interaction_data(ag_chains, ab_chains, return_bio_pdb_aas=False, atom_radius=4):
    
    epitope_data, paratope_data = [], []

    for ag_chain in ag_chains:
        for ab_chain in ab_chains:
            ab_epi_para_res = atom_neighbourhead_search_return_res(NeighborSearch( list(ag_chain.get_atoms()) ), list(ab_chain.get_atoms()), atom_radius=atom_radius)
            epitope_d, paratope_d = get_epitope_paratope_data(ab_epi_para_res, ag_chain, ab_chain, return_bio_pdb_aas=return_bio_pdb_aas)
            epitope_data.extend(epitope_d)
            paratope_data.extend(paratope_d)

    return epitope_data, paratope_data


def atom_neighbourhead_search_return_res(search_object, search_atoms, atom_radius=4):
    paired_interacting_residues = list()

    for search_atom in search_atoms:
        interact_res = search_object.search(search_atom.coord, radius = 4, level="R")

        if interact_res:
            search_residue = search_atom.get_parent()
            if len(interact_res) == 1:
                paired_interacting_residues.append([interact_res[0], search_residue] )
            elif len(interact_res) > 1:
                for int_r in interact_res: paired_interacting_residues.append([int_r, search_residue])

    return paired_interacting_residues


def get_epitope_paratope_data(paired_residues, ag_chain, lc_or_hc, return_bio_pdb_aas = False):
    """
    
    """
    ag_id, ab_id = ag_chain.get_id(), lc_or_hc.get_id()
    antigen_residues = list(ag_chain.get_residues())
    antibody_residues = list(lc_or_hc.get_residues())


    epitope_data = []
    paratope_data = []

    ag_seq = write_pdb_res_to_seq(antigen_residues)
    lc_or_hc_seq = write_pdb_res_to_seq(antibody_residues)
    #create dict with indexes.
    ag_lookup = {antigen_residues[i]:i for i in range(len(antigen_residues))}
    ab_lookup = {antibody_residues[i]:i for i in range(len(antibody_residues))}

    for pair in paired_residues:
        epi_res = pair[0]
        para_res = pair[1]
        try:
            epi_res_idx = ag_lookup[epi_res] 
            para_res_idx = ab_lookup[para_res]
    
        except ValueError:
            sys.exit(f"Could not find paired residues in bio pdb residue list: {pair}")

        if not return_bio_pdb_aas:
            #get amino acid name'
            epi_res = AA3to1_DICT[epi_res.get_resname()]
            para_res = AA3to1_DICT[para_res.get_resname()]

        epitope_data.append((epi_res, ag_id, epi_res_idx) )
        paratope_data.append( (para_res, ab_id, para_res_idx) )
        
    return epitope_data, paratope_data

def write_pdb_res_to_seq(residues):
    """
    residues: Bio PDB residues
    """
    AA_seq = str()
    get_and_remove_heteroatoms(residues)

    for residue in residues:
        try:
            aa = AA3to1_DICT[residue.get_resname()]
        #when residue is something nonstandard
        except KeyError:
            print(aa)
            print("Non-standard amino acid detected")
            aa = "X"

        AA_seq += aa

    return AA_seq




