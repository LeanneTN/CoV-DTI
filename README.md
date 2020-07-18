# NeoDTI for COVID-19
A data-driven drug repositioning framework discovered a potential therapeutic agent targeting COVID-19
 [(biorxiv)](https://www.biorxiv.org/content/10.1101/2020.03.11.986836v1.abstract).


# Requirements
## Hardware Requirements
Our code requires GPU to faciliate the neural network training. The GPU we used to train the model is Nvidia GeForce GTX 1080 Ti.

## Software Requirements
Our code is tested on Ubuntu 16.04.6 LTS with the following software dependencies:
* Python 2.7 
* Tensorflow (tested on version 1.5.0)
* numpy (tested on version 1.16.4)
* sklearn (tested on version 0.20.3)
* tflearn

# Installation Guide
## Install the dependencies
Install Python 2.7
```
sudo apt update
sudo apt upgrade
sudo apt install python2.7 python-pip
 ```
Install numpy
```
pip install numpy==1.16.4
 ```
Install sklearn
```
pip install scikit-learn==0.20.3
```
Install tensorflow
```
pip install tensorflow-gpu==1.5.0
 ```
Install tflearn
```
pip install tflearn
```

## Run the code
To run our experiments:
1. Unzip data.rar in ./data.
2. Execute <code>python NeoDTI_for_COVID19.py</code> to run the 10 fold cross validation results of NeoDTI as well as the prediction results. Options are:  
`-d: The embedding dimension d, default: 512.`  
`-n: Global norm to be clipped, default: 1.`  
`-k: The dimension of project matrices, default: 256.`  
`-r: The number of 10 fold cross validation to run, default: 10.`  
`-e: The epoch number for model training, default: 5000.`  
`-l: L2 regularization strength, default: 0.`  

# Data description
* drug.txt: list of drug names.
* protein.txt: list of protein names.
* disease.txt: list of disease names.
* se.txt: list of side effect names.
* drug_dict_map: a complete ID mapping between drug names and DrugBank ID.
* protein_dict_map: a complete ID mapping between protein names and UniProt ID.
* mat_drug_se.txt : Drug-SideEffect association matrix.
* mat_protein_protein.txt : Protein-Protein interaction matrix.
* mat_drug_drug.txt : Drug-Drug interaction matrix.
* mat_protein_disease.txt : Protein-Disease association matrix.
* mat_drug_disease.txt : Drug-Disease association matrix.
* mat_protein_drug.txt : Protein-Drug interaction matrix.
* mat_drug_protein.txt : Drug-Protein interaction matrix.
* Similarity_Matrix_Drugs.txt : Drug & compound similarity scores based on chemical structures of drugs (\[0,708) are drugs, the rest are compounds).
* Similarity_Matrix_Proteins.txt : Protein similarity scores based on primary sequences of proteins.
* mat_drug_protein_homo_protein_drug.txt: Drug-Protein interaction matrix, in which DTIs with similar drugs (i.e., drug chemical structure similarities > 0.6) or similar proteins (i.e., protein sequence similarities > 40%) were removed (see the paper).
* mat_drug_protein_drug.txt: Drug-Protein interaction matrix, in which DTIs with drugs sharing similar drug interactions (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_sideeffect.txt: Drug-Protein interaction matrix, in which DTIs with drugs sharing similar side effects (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_disease.txt: Drug-Protein interaction matrix, in which DTIs with drugs or proteins sharing similar diseases (i.e., Jaccard similarities > 0.6) were removed (see the paper).
* mat_drug_protein_unique: Drug-Protein interaction matrix, in which known unique and non-unique DTIs were labelled as 3 and 1, respectively, the corresponding unknown ones were labelled as 2 and 0 (see the paper for the definition of unique). 
* mat_compound_protein_bindingaffinity.txt: Compound-Protein binding affinity matrix (measured by negative logarithm of _Ki_).

All entities (i.e., drugs, compounds, proteins, diseases and side-effects) are organized in the same order across all files. These files: drug.txt, protein.txt, disease.txt, se.txt, drug_dict_map, protein_dict_map, mat_drug_se.txt, mat_protein_protein.txt, mat_drug_drug.txt, mat_protein_disease.txt, mat_drug_disease.txt, mat_protein_drug.txt, mat_drug_protein.txt, Similarity_Matrix_Proteins.txt, are extracted from https://github.com/luoyunan/DTINet.



# Contacts
If you have any questions or comments, please feel free to email Fangping Wan (wanfangping92[at]gmail[dot]com) and/or Jianyang Zeng (zengjy321[at]tsinghua[dot]edu[dot]cn).

