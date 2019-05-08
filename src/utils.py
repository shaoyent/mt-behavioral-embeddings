import re 

from itertools import islice
from math import ceil

def window(seq, size=4, step=2, pad='<pad>'):
    l = len(seq) 
    d = size - step 
    n = (ceil((l - d) / step) * step + d - l)
    pad_seq = [pad for _ in range(n)]

    pad_seq.extend(seq)
    seq = pad_seq
    l = len(seq)
    print(seq)
    i = 0
    while i+size <= l :
        res = seq[i:i+size]
        yield res
        i += step
    # assert res[-1] == seq[-1]


def my_tokenizer(sentence) :
    def valid_word(w) :
        return w != "'" and w != "#" and w != "--" and w != ";" and w != "[" and w != "]" and w != "%" and w != "_" and w != ":" and w != "*" and w != "(" and w != ")" and w != "-" and w != "..." and w != ","

    return [ w for w in sentence.split() if valid_word(w) ]
