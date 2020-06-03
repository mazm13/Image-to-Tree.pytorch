from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNCellBase

from .AttTreeModel import AttTreeModel


class MDLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MDLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_prob_lm = 0.5
        self.bias = bias
        self.weight_f1 = Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.weight_f2 = Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.weight_H = Parameter(torch.Tensor(3 * hidden_size, 2 * input_size + 2 * hidden_size))
        if bias:
            self.bias_f1 = Parameter(torch.Tensor(hidden_size))
            self.bias_f2 = Parameter(torch.Tensor(hidden_size))
            self.bias_H = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_f1', None)
            self.register_parameter('bias_f2', None)
            self.register_parameter('bias_H', None)
        # self.dropout = nn.Dropout(self.drop_prob_lm)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def forward(self, input1, input2, hx1=None, hx2=None):
        self.check_forward_input(input1)
        self.check_forward_input(input2)
        self.check_forward_hidden(input1, hx1[0], '[0]')
        self.check_forward_hidden(input1, hx1[1], '[1]')
        self.check_forward_hidden(input2, hx2[0], '[0]')
        self.check_forward_hidden(input2, hx2[1], '[1]')

        H = torch.cat([input1, input2, hx1[0], hx2[0]], dim=1)
        gates = F.linear(H, self.weight_H, self.bias_H)
        ingate, cellgate, outgate = gates.chunk(3, 1)

        ingate = F.sigmoid(ingate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        forgetgate1 = F.linear(torch.cat([input1, hx1[0]], dim=1), self.weight_f1, self.bias_f1)
        forgetgate2 = F.linear(torch.cat([input2, hx2[0]], dim=1), self.weight_f2, self.bias_f2)
        forgetgate1 = F.sigmoid(forgetgate1)
        forgetgate2 = F.sigmoid(forgetgate2)

        next_c = forgetgate1 * hx1[1] + forgetgate2 * hx2[1] + ingate * cellgate
        next_h = outgate * F.tanh(next_c)

        # output = self.dropout(next_h)
        output = next_h
        return output, (next_h, next_c)


class ShowTellTreeCore(nn.Module):
    def __init__(self, opt):
        super(ShowTellTreeCore, self).__init__()
        self.lstm = MDLSTMCell(opt.input_encoding_size, opt.rnn_size)
    
    def forward(self, p_xt, s_xt, p_state, s_state, fc_feats, att_feats, p_att_feats, att_masks):
        output, state = self.lstm(p_xt, s_xt, p_state, s_state)
        return output, state


class ShowTellTreeModel(AttTreeModel):
    def __init__(self, opt):
        super(ShowTellTreeModel, self).__init__(opt)
        self.core = ShowTellTreeCore(opt)
        del self.ctx2att
        del self.att_embed
    
    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        return fc_feats, att_feats, None, att_masks


if __name__ == '__main__':
    data = torch.load('sample_data.pt')
    # fc_feats, att_feats, seqs, att_masks, seqtree, parent_idx, seqtree_mask
    tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['att_masks'],
           data['seqtree'], data['seqtree_idx'], data['seqtree_mask']]
    
    tmp = [_ if _ is None else _.cuda() for _ in tmp]

    fc_feats, att_feats, seqs, att_masks, seqtree, parent_idx, seqtree_mask = tmp
    att_feats, att_masks = None, None 
    parent_idx = parent_idx.long()
    seqtree = seqtree.long()

    import argparse
    opt = argparse.Namespace()
    opt.vocab_size = 9185
    opt.max_seqtree_length = 40
    opt.input_encoding_size = 512
    opt.rnn_size = 512
    opt.drop_prob_lm = 0.8
    opt.fc_feat_size = 2048
    opt.att_feat_size = 2048
    opt.att_hid_size = 512

    model = ShowTellTreeModel(opt)
    model = model.cuda()

    print(model)

    out = model(fc_feats, att_feats, seqs, att_masks, seqtree, parent_idx, seqtree_mask)
    print(out.size())
