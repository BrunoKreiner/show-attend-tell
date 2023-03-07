# show-attend-tell
Implementation and testing of the paper "Show, attend and tell"

## Installation 

Use

```
conda create --name <env-name> --file requirements.txt
conda activate <env-name>
```

to install packages.

Finally, to use the tokenizer, use

```
python -m spacy download en_core_web_sm
```

in a command line to download the english tokenizer from spacy.

To install torch use the following link to get started: https://pytorch.org/get-started/locally/

TO install cuda for GPU training: https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html

For wandb use, use

```
```

## Project Structure

```
├── all_captions                    - flickr8k captions
├── all_images                      - flickr8k images
├── models                          - contains models amd their info in csv file format using timestamps
├── notebooks                       - contains jupyter notebooks 
│   ├── train.ipynb                 - contains notebook for loading tokenizer, loading and preparing data, loading model and train loop
│   ├── analysis.ipynb              - contains notebook for analysis of show-attend-tell model
├── reports                         - contains figures and texts
│   └── figures
└── src                             - contains helper/utility functions
│   └── utils.py                    - contains code for encoder, decoder, attention model, dataset/dataloader, collate function, show img with caption etc...
└── README.md
└── requirements.txt

```

# Trained models

## Weird-attention
        attention_weights = torch.tanh(features + alpha.unsqueeze(2)) #deterministic soft attention 
        attention_weights = attention_weights.sum(dim=1)

## Small Parameters:
  * Attention Dimension = 128
  * Decoder Dimension 256

## Big Parameters:
  * Attention Dimension = 256
  * Decoder Dimension = 512

## Biggest Parameters:
  * Attention Dimension = 1024
  * Decoder Dimension = 1024

Decoder Dimension stays the same


# Important Functions and Take-Home-Message

* Gating scalar Beta in Chapter 4.2.1 not implemented

## nn.Embedding

## nn.Linear

    * Applies a linear transformation to the incoming data y = x * W^T + b
    * behind the scenes:
        * y = x.matmul(m.weight.t()) + m.bias  #y = x*W^T + b
    * Links:
        - https://stackoverflow.com/questions/54916135/what-is-the-class-definition-of-nn-linear-in-pytorch

# Questions

- What is the purpose of initializing the hidden state and cell state by the average value in each filter of the CNN (2048 filters)
