from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

from misc import utils
from graph_utils import utils as gutils


# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
parser.add_argument('--data_path', type=str, default='sampled_data/valdata.pt')
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
opt = parser.parse_args()

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_treelabel_h5', 'input_json', 'input_tree_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab']

word_to_ix = {v: k for k, v in vocab.items()}

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()

data = torch.load('sampled_data/testdata.pt')
tmp = [data['fc_feats'][:1], data['att_feats'][:1], data['att_masks']]
tmp = [_ if _ is None else _.cuda() for _ in tmp]

fc_feats, att_feats, att_masks = tmp
time_start = time.time()
with torch.no_grad():
    # for tpt in [0.5, 1.0, 2.0]:
    #     print("sampled by sample with temperature {}:".format(tpt))
    #     (seq, seq_idx, seqLen), seq_logprobs = model(fc_feats, att_feats, att_masks, opt={'sample_method': 'sample', 'temperature': tpt}, mode='sample')
    #     sent = gutils.decode_sequence(vocab, seq, seq_idx, seqLen)
    #     print('\n'.join(sent))
    (seq, seq_idx, seqLen), seq_logprobs = model(fc_feats, att_feats, att_masks, opt={'sample_method': 'greedy', 'beam_size': 1}, mode='sample')
    sent = gutils.decode_sequence(vocab, seq, seq_idx, seqLen)
    for b in range(seq.shape[0]):
        seq_len  = seqLen[b]
        seq_b = seq[b,:seq_len]
        seq_logprobs_b = seq_logprobs[b,:seq_len]
        p = seq_logprobs_b.gather(1, index=seq_b.unsqueeze(1)).sum()
        print('{} (p={} | len={})'.format(sent[b], p, seq_len))
    print('--' * 10)
    print("sampled by beam search:")
    (seq, seq_idx, seqLen), seq_logprobs = model(fc_feats, att_feats, att_masks, opt={'sample_method': 'greedy', 'beam_size': 2, 'length_penalty': 'avg_0', 'suppress_EOB_factor': 2.0}, mode='sample')
    sent = gutils.decode_sequence(vocab, seq, seq_idx, seqLen)
    print('\n'.join(sent))
    for _s in model.done_beams:
        for __s in _s:
            # seq_len = __s['seq'].shape[0]
            seq_len = __s['seqLen'].item()
            print("seq_len: {}".format(seq_len))
            __ss = gutils.decode_sequence(vocab, __s['seq'].unsqueeze(0), __s['seq_idx'].unsqueeze(0), [seq_len])
            print('{} (p={} | len={})'.format(__ss[0], __s['p'], seq_len))
            # print(__s['logps'].gather(1, index=__s['seq'].unsqueeze(1)).sum())
    print('--' * 10)
