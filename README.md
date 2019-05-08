# Behavioral Sentence Embeddings using Unsupervised Online Multitask Learning
This repo contains the code to extract embeddings from our work

[Unsupervised Online Multitask Learning of Behavioral Sentence Embeddings](https://arxiv.org/pdf/1807.06792.pdf)

_Shao-Yeng Tseng, Brian Baucom, and Panayiotis Georgiou_

## Abstract 
The learning of sentence embeddings is an effort to obtain meaningful vector
representations of sentencesthat achieve fidelity in downstream tasks. This
line of work evolved from word embeddings which weretrained in an unsupervised
manner using large-scale corpora.  Current art, however, has shown thatsentence
embeddings trained using supervised techniques often outperformed those trained
without.While many works However, the transfer of embeddings in-domain is still
required.  This transfer in-domain in the final step of embeddings is a
round-about way of doing things. In this work we present amultitask paradigm
for unsupervised learning of sentence embeddings which simultaneously
addressesdomain adaption. We strive to combine the simplicity of using abundant
unsupervised data with transferlearning by introducing an online multitask
objective We show that embeddings generated through thisprocess increase
performance in subsequent domain-relevant tasks. We target affective tasks such
asemotion recognition and behavior analysis and compare our results with
state-of-the-art general-purposesupervised sentence embeddings.

# Requirements
``` bash
spacy
numpy
scipy
sklearn
torch==0.4.0
torchtext
six
dill
https://github.com/IBM/pytorch-seq2seq.git
```

## Usage
### Download checkpoint
``` bash
cd run/
./download_checkpoints.sh
```

### Download spaCy English module
``` bash
python -m spacy download en
```

### Extract embeddings
``` bash
cd run/
./extact_embeddings.sh TEXTLIST
```
where `TEXTLIST` is a file with paths to text files. 

### Arguments
``` bash
python beh_embedding_multi_label.py
  --extract-list EXTRACT_LIST 	List of files to extract sentence embeddings from.
  --out-dir OUT_DIR     			Path to save embeddings
  --numpy               			Save embeddings as numpy array.
```



