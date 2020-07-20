# NNassc
Adenylation domain substrate specificity prediction

Neural Network based adenylation domain substrate specificity prediction (NNassc).
Substrate specificity prediction for fungal adenylation domains. Fungal NRPS code residues are used as an input and list of substrates is given as an output.  

# Installation

Installation of python version >=3.7 

python3.7 

dependencies - sklearn, rdkit, keras, tensorflow

Installation of dependencies could be either done by pip or conda.

```
conda install scikit-learn
conda install -c conda-forge rdkit
conda install -c conda-forge keras
conda install -c conda-forge tensorflow
```

Creation of python (virtual) environment to run NNassc:

```
conda create --name NNassc_env
```

Activation of the environment:

```
conda activate NNassc_env
```

Deactivation of the environment:

```
conda deactivate NNassc_env
```

# Usage - example input and output

USAGE:

```
python3.7 NNassc_main.py -i "DPRHFVMRA"
```

INPUT: 

9 NRPS code residues: DPRHFVMRA

OUTPUT: 

Rank	Tanimoto	Substarte
1.		1.0 		2-Aminoadipicacid
2.		0.8 		ornithine
3.		0.791		2-Aminobutyricacid
4.		0.778		Glutamine
5.		0.748		Alanine


# Acknowledgements

Financial support for this research work was provided by CRC 1127 ChemBioSys and Hans Knöll Instittue, Jena, Germany and German Centre for Biodiversity Research (iDiv), Leipzig, Germany.

# Contact

Please contact me if you have any further questions: sagargore26@gmail.com 


# LICENSE

NNassc is licenced under GNU General Public License v3.0. Please check LICENCE file for further information.

