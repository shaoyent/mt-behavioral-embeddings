from __future__ import print_function

import os
import argparse
import logging

import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
from torchtext import data

import seq2seq
from seq2seq.models import DecoderRNN
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
# from seq2seq.trainer import SupervisedTrainer
# from seq2seq.models import EncoderRNN
# from seq2seq.loss import Perplexity

from pytorchSeq2seq.models import EncoderConvRNN as EncoderRNN
from pytorchSeq2seq.trainer import MultiLabelTrainer
from pytorchSeq2seq.seq2seq import Seq2seqBeh as Seq2seq
from pytorchSeq2seq.loss import Perplexity

import spacy
from utils import my_tokenizer, window

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()
parser.add_argument('--train-path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev-path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--ckpt-dir', action='store', dest='ckpt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load-checkpoint', action='store', dest='load_checkpoint',
                    help='(Evaluation mode) The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
# parser.add_argument('--vocab_size', action='store', dest='vocab_size', default=50000, type=int,
#                     help='Size of the vocabulary.')
parser.add_argument('--epochs', action='store', dest='epochs', default=5, type=int,
                    help='Number of epochs for training.')
parser.add_argument('--postfix', action='store', dest='postfix', default='',
                    help='Prefix to name folder.')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

subparsers = parser.add_subparsers(dest='mode')

parser_extract = subparsers.add_parser('extract')
parser_extract.add_argument('--extract-list', action='store', dest='extract_list', required=True,
                    help='List of files to extract sentence embeddings from.')
parser_extract.add_argument('--out-dir', action='store', dest='out_dir', default='./',
                    help='Path to save embeddings')
parser_extract.add_argument('--numpy', action='store_true', dest='numpy', default=False,
                    help='Save embeddings as numpy array.')


opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.ckpt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.ckpt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab

    nlp = spacy.load("en", disable=["parser", "tagger", "ner", "textcat"])
    encoder = seq2seq.encoder
    
    if opt.mode == 'extract' : 
        # Extract sentence embeddings from a list of files
        embedding_path = os.path.join(opt.out_dir, '.'.join(opt.extract_list.split('/')[-1].split('.')[:-1]))
        if not os.path.exists(embedding_path) : 
            os.makedirs(embedding_path)

        with open(opt.extract_list,'r') as fh : 
            for line in fh : 
                filepath = line.strip()
                if not os.path.exists(filepath) : 
                    print('File not found : {}'.format(filepath))
                    continue

                filename = filepath.split('/')[-1]

                X = []
                with open(filepath,'r', encoding='utf-8') as fin : 
                    for line in fin : 
                        doc = nlp(line.strip())
                        seq_str = " ".join([ '<name>' if t.text =='propn' else t.text for t in doc ] )
                        src_seq = my_tokenizer(seq_str)

                        if len(src_seq) == 0 :
                            src_seq = "<unk>"
                        src_id_seq = torch.autograd.Variable(torch.LongTensor([input_vocab.stoi[tok] for tok in src_seq]), volatile=True).view(1, -1)
                        if torch.cuda.is_available():
                            src_id_seq = src_id_seq.cuda()

                        encoder_outputs, encoder_hidden = encoder(src_id_seq, [len(src_seq)])
                        try : 
                            if 'HOE' in opt.postfix : 
                                X.append(encoder_hidden.view(1,-1).data.tolist()[0])
                            elif 'BIE' in opt.postfix : 
                                X.append(encoder_hidden[-2:].view(1,-1).data.tolist()[0])
                            else :
                                X.append(encoder_hidden[-2:].view(1,-1).data.tolist()[0])
                                # fout.write(' '.join([ str(x) for x in encoder_hidden[-1][-1].data]))
                        except : 
                            print('***ERROR embedding_output')
                            import ipdb
                            ipdb.set_trace()

                if opt.numpy :
                    with open( os.path.join(embedding_path, filename.replace('.txt','.npy')), 'wb') as fout :
                        np.save(fout, np.array(X))
                else :
                    with open( os.path.join(embedding_path, filename), 'w') as fout :
                        for x in X : 
                            print(' '.join([str(i) for i in x]), file=fout)
                     
                print('Parsed {}'.format(filename))

    else :
        # Interactive mode
        predictor = Predictor(seq2seq, input_vocab, output_vocab)

        while True:
            seq_str = raw_input("Type in a source sequence:")
            doc = nlp(seq_str.strip())
            seq_str = " ".join([ t.text for t in doc ] )
            seq = my_tokenizer(seq_str)

            print(predictor.predict(seq))

            src_id_seq = torch.autograd.Variable(torch.LongTensor([input_vocab.stoi[tok] for tok in seq]), volatile=True).view(1, -1)
            if torch.cuda.is_available():
                src_id_seq = src_id_seq.cuda()

            encoder_outputs, encoder_hidden = encoder(src_id_seq, [len(seq)])
            print(encoder_hidden[-1][0][0:20])
            print(encoder_outputs[-1][0][0:20])


else:
    # Initialize model from expt dir
    model_name = opt.ckpt_dir.split('/')[-1]
    configs = model_name.split('_')
    assert len(configs) >= 3

    vocab_size = int(configs[0])
    num_layers = int(configs[1])
    hidden_size = int(configs[2])

    # Prepare dataset
    src = SourceField(tokenize=my_tokenizer)
    tgt = TargetField(tokenize=my_tokenizer)
    # beh = data.Field(sequential=True, tensor_type=torch.DoubleTensor, batch_first=True, use_vocab=False, postprocessing=data.Pipeline(lambda x,y,z: [float(n) for n in x]))
    beh = data.Field(sequential=True, tensor_type=torch.LongTensor, batch_first=True, use_vocab=False, postprocessing=data.Pipeline(lambda x,y,z: int(np.argmax([ float(n) for n in x ]))))

    max_len = 50
    def len_filter(example):
        try : 
            return len(example.src) <= max_len and len(example.tgt) <= max_len and len(example.src) > 1 and len(example.tgt) > 1
        except :
            print('***ERROR: len_filter')
            import ipdb
            ipdb.set_trace()
            return False

    print('Initializing dataset')
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('src', src), ('tgt', tgt), ('beh', beh)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )

    if not os.path.exists(opt.ckpt_dir) :
        os.makedirs(opt.ckpt_dir)

    if opt.resume :
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(opt.ckpt_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        src.vocab = resume_checkpoint.input_vocab
        tgt.vocab = resume_checkpoint.output_vocab
    else :
        print('Building vocab')
        #src.build_vocab(train, max_size=50000)
        #tgt.build_vocab(train, max_size=opt.vocab_size, vectors='glove.840B.300d')
        if hidden_size == 300 :
            vectors = 'glove.42B.300d'
        elif hidden_size == 100 : 
            vectors = 'glove.6B.100d'
        else : 
            vectors = None 

        tgt.build_vocab(train, max_size=vocab_size, vectors=vectors)
        src.vocab = tgt.vocab
        input_vocab = src.vocab
        output_vocab = tgt.vocab

        vocab = tgt.vocab

        with open(os.path.join(opt.ckpt_dir, 'vocab.txt'), 'w') as f :
            for i, w in enumerate(input_vocab.itos) :
                f.write('{} {}'.format(str(w), i))
                f.write('\n')

    # with open('output_vocab.txt', 'w') as f :
    #     for i, w in enumerate(output_vocab.itos) :
    #         f.write('{} {}'.format(str(w), i))
    #         f.write('\n')

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        print('Vocab : {} Layers : {} Size : {}'.format(vocab_size, num_layers, hidden_size)) 
        # hidden_size=100
        # num_layers=3
        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size, dropout_p=0.2,
                             n_layers=num_layers, bidirectional=bidirectional, variable_lengths=True, vectors=vocab.vectors)
        decoder = DecoderRNN(len(tgt.vocab), max_len, (hidden_size * 2) if bidirectional else hidden_size,
                             n_layers=num_layers, dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            # param.data.uniform_(-0.08, 0.08)
            param.data.normal_(0.0, 0.1)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        optimizer = Optimizer(torch.optim.SGD(seq2seq.parameters(), lr = 0.05, momentum=0.9), max_grad_norm=5)
        scheduler = StepLR(optimizer.optimizer, 1)
        optimizer.set_scheduler(scheduler)

    # train
    t = MultiLabelTrainer(loss=loss, batch_size=64,
                          checkpoint_every=10000,
                          print_every=100, ckpt_dir=opt.ckpt_dir)

    print('Start training')
    seq2seq = t.train(seq2seq, train,
                      num_epochs=opt.epochs, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.9,
                      resume=opt.resume)


print('Done')


