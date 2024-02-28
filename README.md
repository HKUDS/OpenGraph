# OpenGraph: Towards Open Foundation Models for Graph

This project presents OpenGraph, a foundation graph model with the capabilities of <b><i>generalizing to unseen graph data</i></b> that significantly differs from the trianing instances.

<img src='intro.png' width=60% />

To achieve this goal, OpenGraph addresses several key technical challenges:
- We propose a unified graph tokenizer to adapt our graph model to generalize well on unseen graph data, even when the underlying graph properties differ significantly from those encountered during training. 
- We develop a scalable graph transformer as the foundational encoder, which effectively captures node-wise dependencies within the global topological context. 
- We introduce a data augmentation mechanism enhanced by a large language model (LLM) to alleviate the limitations of data scarcity in real-world scenarios.

<img src='framework.png' />

Extensive experiments validate the effectiveness of our framework. By adapting OpenGraph to new graph characteristics and comprehending the nuances of diverse graphs, our approach achieves remarkable zero-shot graph learning performance across various settings and domains.

## Environment Setup
You need to unzip some of the data files in `datasets/`. Our experiments were conducted with the following package versions:
* python==3.10.13
* torch==1.13.0
* numpy==1.23.4
* scipy==1.9.3

## Usage
#### To reproduce the test performance reported in the paper, run the following command lines:
```
cd link_prediction/
python main.py --load pretrn_gen1 --epoch 0 # test on OGBL-Collab, ML-1M, ML-10M
python main.py --load pretrn_gen0 --tstdata amazon-book --epoch 0 # test on Amazon-Book
python main.py --load pretrn_gen2 --tstdata ddi --epoch 0 # test on OGBL-ddi
cd ../node_classification/
python main.py --load pretrn_gen1 --tstdata cora # test on Cora
python main.py --load pretrn_gen1 --tstdata citeseer # test on Citeseer
python main.py --load pretrn_gen1 --tstdata pubmed # test on Pubmed
```
<img src='performance.png' />

#### To re-pretrain OpenGraph by yourself, run the following command lines:
```
cd ../link_prediction/
python main.py --save pretrn_gen1
python main.py --trndata gen0 --tstdata amazon-book --save pretrn_gen0
python main.py --trndata gen2 --tstdata ddi --save pretrn_gen2
```

#### To explore pretraining with multiple different pre-training and testing datasets, modify `trn_datasets` and `tst_datasets` in line 241 of `link_prediction/main.py`.

## Graph Data Generation
To be completed.
