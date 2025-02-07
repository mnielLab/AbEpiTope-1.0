# AbEpiTope-1.0
AbEpiTope-1.0 is a computational tool that features two scores: AbEpiScore-1.0, predicting the accuracy of modelled AbAg interfaces, and AbEpiTarget-1.0, for selecting the antibody most likely targeting a given antigen. Both use the pretrained inverse folding model, ESM-IF1. As input, both models expect modelled antibody-antigen interfaces (PDB/CIF of AlphaFold or experimentally solved structures).

## License
AbEpiTope-1.0 was developed by the Health Tech section at Technical University of Denmark (DTU). The code and data can be used freely by academic groups for non-commercial purposes. If you plan to use these tools for any for-profit application, you are required to obtain a separate license (contact Morten Nielsen, morni@dtu.dk).

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







