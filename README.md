# Behavioral Sentence Embeddings using Unsupervised Online Multitask Learning
This repo contains the code to extract embeddings from our work

[Unsupervised Online Multitask Learning of Behavioral Sentence Embeddings](https://arxiv.org/pdf/1807.06792.pdf)

_Shao-Yeng Tseng, Brian Baucom, and Panayiotis Georgiou_

## Abstract 
Appropriate embedding transformation of sentences can aid in downstream tasks such as NLP and emotion and behavior analysis. 
Such efforts evolved from word vectors which were trained in an unsupervised manner using large-scale corpora. 
Recent research, however, has shown that sentence embeddings trained using in-domain data or supervised techniques, often through multitask learning, perform better than unsupervised ones.
Representations have also been shown to be applicable in multiple tasks, especially when training incorporates multiple information sources.
In this work we aspire to combine the simplicity of using abundant unsupervised data with transfer learning by introducing an online multitask objective.
We present a multitask paradigm for unsupervised learning of sentence embeddings which simultaneously addresses domain adaption.
We show that embeddings generated through this process increase performance in subsequent domain-relevant tasks.
We evaluate on the affective tasks of emotion recognition and behavior analysis and compare our results with state-of-the-art general-purpose supervised sentence embeddings.
Our unsupervised sentence embeddings outperform the alternative universal embeddings in both identifying behaviors within couples therapy and in emotion recognition.

## Requirements
``` bash
spacy
numpy
scipy
sklearn
torch==0.4.1
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
  --out-dir OUT_DIR     	Path to save embeddings
  --numpy               	Save embeddings as numpy array.
```



