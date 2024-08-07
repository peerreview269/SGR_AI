
# Coding Small Group Communication: Transformer and RNN Models with Context

This repository provides instructions and scripts for reproducing RNN and Transformer models with context windows using the Silicone datasets (https://huggingface.co/datasets/eusip/silicone). The project aims to facilitate the coding of small group communication dynamics.

## Prerequisites

Before running the scripts, make sure you have the following dependencies installed:

- Pytorch
- Transformers
- scikit-learn
- Datasets

You can install these packages using pip:

```bash
pip install torch transformers scikit-learn datasets
```

## Installation and Setup

Clone the repository to your local machine or just donwload the files. Whatever you want:

```bash
git clone https://github.com/peerreview269/SGR_AI
cd your-repo-name
```

## Running the Models

To train the model on different datasets, simply run the corresponding files:

### BERT Model with Context

```bash
python run_BERT_with_context.py
```

### RNN Model (LSTM)

```bash
python run_RNN_LSTM.py
```

## Selecting the Dataset

In each script, choose the appropriate dataset in the `Load Dataset` portion of the file.

## Running the AMI using BERT with Context

```bash
python AMI.py
```
### RNN Model (LSTM) using AMI

```bash
python AMI_RNN.py
```

## Configuring Context in BERT Models

For different context configurations in the BERT models, modify the `custom_configurations` line in the script. You can modify type of context (previous, subsequent, both) and 
the window size (integer).

