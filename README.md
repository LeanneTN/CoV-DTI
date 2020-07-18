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
1. Unzip data.rar or data_original.rar in ./data. The latter one corresponds to the original dataset we used in the manuscript while the former one added new virus protein-human protein interaction data [Gordon, D.E., Jang, G.M., Bouhaddou, M. et al. A SARS-CoV-2 protein interaction map reveals targets for drug repurposing. Nature (2020)] that further improved the prediction performance. 

2. <code> cd ./src </code>, execute <code>python NeoDTI_for_COVID19.py</code> to run the 10 fold cross validation results of NeoDTI as well as the prediction results. Options are:  
`-d: The embedding dimension d, default: 512.`  
`-n: Global norm to be clipped, default: 1.`  
`-k: The dimension of project matrices, default: 256.`  
`-r: The number of 10 fold cross validation to run, default: 10.`  
`-e: The epoch number for model training, default: 5000.`  
`-l: L2 regularization strength, default: 0.`  
The results are stored in ./output.

# Data description
* drug_iddict: python dictionary, key: drug InChI; value: index number.
* Drug_simi_net.npy: Drug structure similarity matrix (tanimoto similarity of Morgan fingerprints).
* new_all_human_seq_iddict.pkl: python dictionary, key: human protein ID; value: index number.
* new_all_human_seq_seqdict.pkl: python dictionary, key: human protein ID; value: protein sequence.
* new_all_human_seq.npy: Human protein sequence similarity matrix (normalized Smith-Waterman alignment scores).
* PPI_net.npy: Human protein-protein interaction matrix.

* all_seq_virus_seqdict.pkl: python dictionary, key: virus protein ID; value: index number.
* all_seq_virus_iddict.pkl: python dictionary, key: virus protein ID; value: index number.
* all_seq_virus.npy: Virus protein sequence similarity matrix (normalized Smith-Waterman alignment scores).

* VHI_net.npy: Virus protein-human protein interaction matrix.
* VDTI_net.npy: Drug-virus protein interaction matrix.
* HDTI_net.npy: Drug-human protein interaction matrix.

All entities (i.e., drugs, human and virus proteins) are ordered by their indices. 



# Contacts
If you have any questions or comments, please feel free to email Fangping Wan (wanfangping92[at]gmail[dot]com) and/or Jianyang Zeng (zengjy321[at]tsinghua[dot]edu[dot]cn).

