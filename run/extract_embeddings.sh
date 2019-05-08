#! /bin/bash

TEXTLIST=textlist.txt
if [ ! -z $1 ] ; then
    TEXTLIST=$1
else 
	echo "Usage: extract_embeddings.sh [TEXTLIST]"
fi


# Best negative model
model=30000_3_100_m2
ckpt=2018_01_30_13_05_32

# Best IEMOCAP model
# model=30000_3_300_m5
# ckpt=2018_03_03_17_23_00

CUDA_VISIBLE_DEVICES=0 python ../src/beh_embedding_multi_label.py \
    --ckpt-dir ../checkpoints/$model \
    --load-checkpoint $ckpt \
    --postfix _HOE \
    extract \
    --extract-list $TEXTLIST \
    --out-dir ./embeddings 
