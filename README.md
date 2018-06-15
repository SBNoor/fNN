# fNN

fNN is a command line tool for predicting secondary structure of proteins using their primary sequence, multiple sequence alignment data (in FASTA format), or position-specific scoring matrix (PSSM).

## System Speciﬁcations 
For the tool to work properly, following packages are required:

• Python 3.6 : https://anaconda.org/anaconda/python

• Keras : https://anaconda.org/conda-forge/keras

• Scikit-learn : https://anaconda.org/anaconda/scikit-learn


## Getting started

In order to download fNN, you should clone the repository via the commands

```
git clone https://github.com/SBNoor/fNN.git

cd fNN
```

Once the above process has completed, you can run

```
python tool.py -h
```

to list all of the command-line options. If this command fails it means that something went wrong during the installation process.

## Structure Prediction

In this section you will predict the secondary structure of a protein using its primary sequence. We will assume you have already followed the instructions for downloading Python 3.6, Keras, Scikit-learn and fNN.

The following command will allow you to predict the secondary structure of a protein:

```
python tool.py <file _ name>.fasta
```

Protein structure prediction will take approximately 12 minutes if a single FASTA sequence is given as an input and will take approximately 27 minutes if the input is a multiple sequence alignment. The time is mostly spent on training the neural network.

You can also specify which neural network you want to give your data to. In that case you can use one of the following ﬂags:

**-j JNN** 

This ﬂag will run the neural network described by Qian and Sejnowski, which is a simple one hidden layer feed-forward neural network that requires a single sequence as an input. This neural network will run by default if you provide a single fasta sequence even without the above mentioned ﬂag. The command is :

```
python tool.py -j JNN <file _ name>.fasta
```

**-js MSA** 

This ﬂag will run a standard feed-forward neural network that requires a single sequence as an input but that sequence is generated from a multiple sequence alignment by virtue of majority voting. This neural network is the one that will be used by default if a multiple sequence alignment is provided without a ﬂag. The command is: 

```
python tool.py -js MSA <file _ name>.fasta
```

**-m mNN**

This ﬂag will predict the secondary structure using the network similar to the one explained by Rost and Sander. It is a cascaded neural network whereby the ﬁrst neural network is a sequence - to - structure network and the second one is a structure - to - structure neural network. The command is as follows:

```
python tool.py -m mNN <file _ name>.fasta
```

**-s sNN**

This ﬂag runs a convolutional neural network based on the approach of Liu and Chen. This neural network will run by default if the user enters a PSSM as an input. The command to be entered is:

```
python tool.py -s sNN <file _ name>.pssm
```

You can also have the prediction written to a text ﬁle using the -o ﬂag:

```
python tool.py -o <file _ name>.fasta
```