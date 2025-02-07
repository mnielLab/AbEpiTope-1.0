# AbEpiTope-1.0
AbEpiTope-1.0 is a computational tool that features two scores: AbEpiScore-1.0, predicting the accuracy of modelled AbAg interfaces, and AbEpiTarget-1.0, for selecting the antibody most likely targeting a given antigen. Both use the pretrained inverse folding model, ESM-IF1. As input, both models expect predicted or solved antibody-antigen interfaces in PDB/CIF format. AbEpiTope-1.0 was trained to evaluate predicted AlphaFold structures, and it will not produce good scoring for structures produced by rigid-body docking.

## License
AbEpiTope-1.0 was developed by the Health Tech section at Technical University of Denmark (DTU). The code and data can be used freely by academic groups for non-commercial purposes. If you plan to use these tools for any for-profit application, you are required to obtain a separate license (contact Morten Nielsen, morni@dtu.dk).

## Webserver
AbEpiTope-1.0 is freely available as a web server at [https://services.healthtech.dtu.dk/services/AbEpiTope-1.0.](https://services.healthtech.dtu.dk/services/AbEpiTope-1.0). 

## Installation 
It is important that you follow the steps and do not install a latest pytorch and cudatoolkit version. 
The reason is that we need the installation to be compatible with a Pytorch Geometric.

### Create Conda Environment
```
$ conda create -n inverse python=3.9 ## important that it is python version 3.9
$ conda activate inverse
$ conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch   ## very important to specify pytorch package!
$ conda install pyg -c pyg -c conda-forge ## very important to make sure pytorch and cuda versions not being changed
$ conda install pip
```
### Install Pip Packages 
```
First, download requirements.txt file. Then,
$ pip install -r requirements.txt #install package dependencies
$ pip install git+https://github.com/mnielLab/AbEpiTope-1.0.git #install source code directly with pip
```
### Usage 
We provide a python code snippet hereunder as well as a notebook (demo_notebook.ipynb) for running AbEpiTope-1.0 on 30 AlphaFold-2.3 predicted strucutures of antibody targeting the PD1-receptor (PDB: 7E9B).
These predicted structures can found under ./abag_exampledata/Cancer. 

## Inputs 

AbEpiTope-1.0 evaluates structure files of antibody-antigen complexes (pdb/cif). These structure files can be solved or predicted structures.
1. Each structure file must include a light and heavy chain or a single-chain variable fragment (scFv), along with one or more antigen chains. **Note:** Scores will not be produced for antibody-antigen structures where an where this is not detected. 
2. The antibody-antigen interface is made up of epitope and paratope residues. We define epitope residues as any antigen residues with at least one heavy atom (main-chain or side-chain) at a distance of 4 Å or less to any light or heavy chain. The corresponding interacting residues on the light or heavy chain are the paratope residues. **Note:** Scores will not be produced if epitope and paratope residues are not detected. By default, thet distance is set at 4 Å, but can be set to custom Angstrom (Å). 

## Example

```python
# imports and static stuff
import torch
from abepitope.main import StructureData, EvalAbAgs
from pathlib import Path
STRUCTUREINPUTS = Path.cwd() / "abag_exampledata" / "Cancer" # directory containing PDB or CIF files (can also be a single PDB/CIF file)
ENCDIR = Path.cwd() / "encodings" # directory for storing ESM-IF1 encodings
TMPDIR = Path.cwd() / "temporary" # directort for storing temporary files 

## create abag interface encodings ##
# encode antibody-antigen complex structure files at default 4 Å distance
data = StructureData()
data.encode_proteins(/path/to/structure(s), /path/to/store/temporary/encodings/, /path/to/store/temporary/stuff)
# or encode antibody-antigen complex structure files at custom distance, such as 4.5 Å distance
data = StructureData()
data.encode_proteins(/path/to/structure(s), /path/to/store/temporary/encodings/, /path/to/store/temporary/stuff, atom_radius=4.5)

## evaluate antibody-antigen complexes ## 

# compute abepiscore-1.0 scores only

# compute abepitarget-1.0 scores only

# compute both scores and store scores in .csv file together with other files 


```

## Run

After this step, antibody-antigen complex interfaces, can be scored with AbEpiTope-1.0. 
```
$ eval_abags = EvalAbAgs(data)
$ eval_abags.predict(/path/to/output/)
```
The resulting 


3. Users can set a custom Angstrom (Å) distance for defining antibody-antigen interfaces.

The default is 4 Å. 
Note: Scores will not be produced for antibody-antigen structures if no epitope and paratope residues are detected at the set Å distance.





