# TSSN

#### Introduction

A Token-based Self-Supervised Network on Traffic Flow data

The source code for the pre-print paper [Masked Token Enabled Pre-training A Task-Agnostic Approach for Understanding Complex Traffic Flow](https://www.techrxiv.org/articles/preprint/Masked_Token_Enabled_Pre-training_A_Task-Agnostic_Approach_for_Understanding_Complex_Traffic_Flow/19134854), which is under review of [IEEE Transactions on Intelligent Transportation Systems](https://mc.manuscriptcentral.com/t-its).

---

The main scripts are in folder `main`, which can be directly ran in Pytorch after cloning this repository and getting all the requried dependencies ready. However, necessary data pre-processing is required to get the same data shape. The pre-processing scheme can be found in our paper.

The model structure is in folder `model`. The code for pretrain and finetune phase is listed in `pretrain` and `finetune` respectively.

The comparison models of LSTM and Transformer are in `LSTM` and `transformer`.



### In this paper(repository),

1) A novel network, i.e. **TSSN**, is proposed for generating an effective task-agnostic model for various downstream tasks on traffic flow data. 

    ![System structure](./structure-eps-converted-to.pdf)

    Meanwhile, a novel pretext task, i.e. **masked token prediction(MTP)**, is designed to provide strong surrogate supervision signals for the pre-training of TSSN.

2) Three types of **downstream tasks**, i.e. TF classification, prediction and completion, are solved by using the representations of tokens created in pre-training model.

### Dataset

The datasets for pre-training and TF prediction and completion tasks are collected from [PeMS](http://pems.dot.ca.gov), and those used for TF classidication task is Seattle Inductive Loop Detector Dataset.
