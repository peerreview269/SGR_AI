# Coding Small Group Communication: Transformer and RNN Models with Context

This repository provides instructions and scripts for reproducing RNN and Transformer models with context using the Silicone datasets. The project aims to facilitate the coding of small group communication dynamics.

## Prerequisites

Before running the scripts, make sure you have the following dependencies installed:

- Pytorch
- Transformers
- sk-learn
- Datasets

You can install these packages using pip:

```bash
pip install torch transformers scikit-learn datasets

## Installation and setup
git clone https://github.com/peerreview269/SGR_AI
cd your-repo-name

## Running the Models
To train the model on different datasets, simply run the corresponding files:
python run_BERT_with_context.py
python run_RNN_LSTM.py

Selecting the Dataset
In each script, choose the appropriate dataset in the Load Dataset portion of the file.

Configuring Context in BERT Models
For different context configurations in the BERT models, modify the custom_configurations line in the script.
