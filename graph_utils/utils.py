import os
from .graph import decode_array

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq, seq_idx, seqLen):
    N, D = seq.size()
    if not isinstance(seqLen, list):
        seqLen = seqLen.tolist()
    out = []
    for i in range(N):
        s = [ix_to_word[str(ix)] if ix > 0 else "<PAD>" for ix in seq[i].tolist()]
        try:
            ttree = decode_array(s[:seqLen[i]], seq_idx[i].tolist()[:seqLen[i]])
        except Exception as e:
            print(seq[i].tolist())
            print(s)
            print(seq_idx[i].tolist())
            print(seqLen[i])
            print(e)
            os._exit(0)
        out.append(str(ttree))
    return out
