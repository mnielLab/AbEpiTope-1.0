# AntiInterNet-1.0 and AntiScout-1.0
The tools, AntInterNet-1.0 and AntiScout-1.0 are computational tools for antibody-specific B-cell epitope prediction, using the pretrained inverse folding model, ESM-IF1.   
They are designed to assess the accuracy of modelled antibody-antigen interfaces (PDB/CIF of AlphaFold or experimentally solved structures) and to select the most likely antibody to bind a given antigen from a pool of candidates, respectively. 

## License
AntiInterNet-1.0 and AntiScout-1.0 are developed by the Health Tech section at Technical University of Denmark (DTU). The code and data can be used freely by academic groups for non-commercial purposes. If you plan to use these tools for any for-profit application, you are required to obtain a separate license (contact Morten Nielsen, morni@dtu.dk).

## Installation 
It is important that you follow the steps and do not install a latest pytorch and cudatoolkit version. 
The reason is that we need the installation to be compatible with a Pytorch Geometric.

### Create Conda Environment
$ conda create -n inverse python=3.9 ## important that it is python version 3.9
$ conda activate 
$ conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch   ## very important to specify pytorch package!
$ conda install pyg -c pyg -c conda-forge ## very important to make sure pytorch and cuda versions not being changed
$ conda install pip

### Install Pip Packages 
$ pip install -r requirements.txt
