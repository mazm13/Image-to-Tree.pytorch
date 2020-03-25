from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNCellBase


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1) # Bx1x...
        x = x.expand(-1, n, *([-1]*len(x.shape[2:]))) # Bxnx...
        x = x.reshape(x.shape[0]*n, *x.shape[2:]) # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()
    
    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_'+mode)(*args, **kwargs)


class AttTreeModel(CaptionModel):
    def __init__(self, opt):
        super(AttTreeModel, self).__init__()
        self.opt = opt
        self.vocab_size = opt.vocab_size
        self.max_seqtree_length = opt.max_seqtree_length
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.hidden_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.num_layers = 1
        self.ss_prob = 0.0

        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(bsz, self.rnn_size),
                weight.new_zeros(bsz, self.rnn_size))
    
    def init_input(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(bsz, self.input_encoding_size)

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks, seqtree, parent_idx, seqtree_mask):
        batch_size = fc_feats.size(0)
        seq_per_img = seq.shape[0] // batch_size
        state = self.init_hidden(batch_size*seq_per_img)

        outputs = fc_feats.new_zeros(batch_size*seq_per_img, seqtree.size(1), self.vocab_size+1)
        
        hidden_states = fc_feats.new_zeros(batch_size*seq_per_img, seqtree.size(1), self.rnn_size)
        cell_states = fc_feats.new_zeros(batch_size*seq_per_img, seqtree.size(1), self.rnn_size)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        if seq_per_img > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = repeat_tensors(seq_per_img,
                [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
            )
        
        for i in range(0, seqtree.size(1)):
            # break if all the sequences end
            if i >= 1 and seqtree[:, i].sum() == 0:
                break

            if i == 0:
                p_xt = p_fc_feats
                p_state = self.init_hidden(batch_size*seq_per_img)
                s_xt = self.init_input(batch_size*seq_per_img)
                s_state = self.init_hidden(batch_size*seq_per_img)
            else:
                p_it = torch.gather(seqtree, dim=1, index=parent_idx[:,i].clone().unsqueeze(1))
                p_it = p_it.squeeze(dim=1)
                p_xt = self.embed(p_it)

                p_idx = parent_idx[:,i].clone()
                p_idx = p_idx.unsqueeze(1).unsqueeze(1).expand(batch_size*seq_per_img, 1, self.hidden_size)
                p_hidden_state = torch.gather(hidden_states.clone(), dim=1, index=p_idx).squeeze(dim=1)
                p_cell_state = torch.gather(cell_states.clone(), dim=1, index=p_idx).squeeze(dim=1)

                p_state = p_hidden_state, p_cell_state

                if i % 3 == 1:
                    s_xt = self.init_input(batch_size*seq_per_img)
                    s_state = self.init_hidden(batch_size*seq_per_img)
                else:
                    s_it = seqtree[:,i-1].clone()
                    s_xt = self.embed(s_it)
                    s_hidden_state = hidden_states[:,i-1].clone()
                    s_cell_state = cell_states[:,i-1].clone()
                    s_state = s_hidden_state, s_cell_state
                
            output, state = self.get_logprobs_state(p_xt, s_xt, p_state, s_state, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks)
            hidden_states[:,i] = state[0]
            cell_states[:,i] = state[1]
            outputs[:,i] = output

        return outputs

    def get_logprobs_state(self, p_xt, s_xt, p_state, s_state, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, output_logsoftmax=1):
        # different Ruotian Luo's implementation, p_xt stands for word embedding
        output, state = self.core(p_xt, s_xt, p_state, s_state, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks)

        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)
        
        return logprobs, state
    
    def sample_next_word(self, logprobs, sample_method, temperature):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
        elif sample_method == 'sample':
            logprobs = logprobs / temperature
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
        else:
            raise NotImplementedError("sample method {} has not been implemented.".format(sample_method))
        return it, sampleLogprobs

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        
        sample_method = opt.get('sample_method', 'greedy')
        beam_size= opt.get('beam_size', 1)
        max_seqtree_length = opt.get('max_seqtree_length', 40)
        temperature = opt.get('temperature', 1.0)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        seq_per_img = 1

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        seqtree = fc_feats.new_zeros(batch_size, max_seqtree_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, max_seqtree_length, self.vocab_size + 1, dtype=torch.float)
        parent_idx = fc_feats.new_zeros(batch_size, max_seqtree_length, dtype=torch.long)

        hidden_states = fc_feats.new_zeros(batch_size*seq_per_img, seqtree.size(1), self.rnn_size)
        cell_states = fc_feats.new_zeros(batch_size*seq_per_img, seqtree.size(1), self.rnn_size)

        counter = fc_feats.new_ones(batch_size, dtype=torch.long)
        seqLen = fc_feats.new_zeros(batch_size, dtype=torch.long)
        all_finished = fc_feats.new_zeros(batch_size, dtype=torch.bool)

        for i in range(max_seqtree_length):

            if all_finished.all():
                break

            if i == 0:
                # TODO: feed root as first input instead of fc_feats.
                p_xt = p_fc_feats
                p_state = self.init_hidden(batch_size*seq_per_img)
                s_xt = self.init_input(batch_size*seq_per_img)
                s_state = self.init_hidden(batch_size*seq_per_img)
            else:
                p_it = torch.gather(seqtree, dim=1, index=parent_idx[:,i].clone().unsqueeze(1))
                p_it = p_it.squeeze(dim=1)
                p_xt = self.embed(p_it)

                p_idx = parent_idx[:,i].clone()
                p_idx = p_idx.unsqueeze(1).unsqueeze(1).expand(batch_size*seq_per_img, 1, self.hidden_size)
                p_hidden_state = torch.gather(hidden_states.clone(), dim=1, index=p_idx).squeeze(dim=1)
                p_cell_state = torch.gather(cell_states.clone(), dim=1, index=p_idx).squeeze(dim=1)

                p_state = p_hidden_state, p_cell_state

                if i % 3 == 1:
                    s_xt = self.init_input(batch_size*seq_per_img)
                    s_state = self.init_hidden(batch_size*seq_per_img)
                else:
                    s_it = seqtree[:,i-1].clone()
                    s_xt = self.embed(s_it)
                    s_hidden_state = hidden_states[:,i-1].clone()
                    s_cell_state = cell_states[:,i-1].clone()
                    s_state = s_hidden_state, s_cell_state

            logprobs, state = self.get_logprobs_state(p_xt, s_xt, p_state, s_state, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, output_logsoftmax=output_logsoftmax)
            hidden_states[:,i] = state[0]
            cell_states[:,i] = state[1]

            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            ifextend = ( (it != self.vocab_size) ).unsqueeze(dim=1).repeat(1,3)
            src = it.new_ones(batch_size, 3, dtype=torch.long) * ifextend * i
            scatter_index = counter.unsqueeze(dim=1).repeat(1, 3) + torch.arange(0, 3, dtype=torch.long, device=it.device)
            scatter_index = torch.min(scatter_index, scatter_index.new_ones(scatter_index.size()) * (self.max_seqtree_length - 1))

            parent_idx.scatter_(dim=1, index=scatter_index, src=src)

            all_finished = all_finished | (counter == i) 
            seqLen = seqLen * all_finished + counter * ~all_finished

            counter.add_((it != self.vocab_size) * 3)
            seqtree[:,i] = it
            seqLogprobs[:,i] = logprobs

        return (seqtree, parent_idx, seqLen), seqLogprobs

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        sample_n = opt.get('sample_n', 3)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)
        seq_per_img = 1

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        
        # first step, feed fc_feats as first word input. t = 0.
        p_xt = p_fc_feats
        p_state = self.init_hidden(batch_size*seq_per_img)
        s_xt = self.init_input(batch_size*seq_per_img)
        s_state = self.init_hidden(batch_size*seq_per_img)
        logprobs, state = self.get_logprobs_state(p_xt, s_xt, p_state, s_state, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks)
        
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = repeat_tensors(beam_size,
            [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
        )

        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        return self.done_beams
    
    def beam_step(self, logprobs, beam_size, t, beam_seq, beam_parent_idx, beam_seq_logprobs, beam_logprobs_sum, state, counter, seqLen, all_finished):
        batch_size = beam_logprobs_sum.size(0)
        vocab_size = logprobs.size(-1)
        unaug_logprobs = logprobs.clone()
        logprobs = logprobs.view(batch_size, -1, vocab_size)
        if t == 0:
            assert logprobs.size(1) == 1
            beam_logprobs_sum = beam_logprobs_sum[:,:1]
        candidate_logprobs = beam_logprobs_sum.unsqueeze(-1) + logprobs  # `(batch_size, beam_size, vocab_size + 1)`

        ys, ix = torch.sort(candidate_logprobs.reshape(candidate_logprobs.shape[0], -1), -1, True)
        ys, ix = ys[:,:beam_size], ix[:,:beam_size]  # `(batch_size, beam_size)`
        beam_ix = ix // vocab_size
        selected_ix = ix % vocab_size
        # print(selected_ix)
        state_ix = (beam_ix + torch.arange(batch_size).type_as(beam_ix).unsqueeze(-1) * logprobs.shape[1]).reshape(-1) # N*b which in Nxb beams
        
        if t > 0:
            # gather according to beam_ix
            assert (beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq)) == beam_seq.reshape(-1, beam_seq.shape[-1])[state_ix].view_as(beam_seq)).all()
            beam_seq = beam_seq.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_seq))
            
            beam_seq_logprobs = beam_seq_logprobs.gather(1, beam_ix.unsqueeze(-1).unsqueeze(-1).expand_as(beam_seq_logprobs))

            beam_parent_idx = beam_parent_idx.gather(1, beam_ix.unsqueeze(-1).expand_as(beam_parent_idx))

        # beam_seq[:,:,t] = selected_ix
        # TODO: why not assign with above may?
        if t == 0:
            beam_seq = torch.cat([beam_seq, ])
        beam_seq = torch.cat([beam_seq, selected_ix.unsqueeze(dim=-1)], dim=-1)

        beam_logprobs_sum = beam_logprobs_sum.gather(1, beam_ix) + \
            logprobs.reshape(batch_size, -1).gather(1, ix)
        assert (beam_logprobs_sum == ys).all()
        _tmp_beam_logprobs = unaug_logprobs[state_ix].reshape(batch_size, -1, vocab_size)
        beam_logprobs = unaug_logprobs.reshape(batch_size, -1, vocab_size).gather(1, beam_ix.unsqueeze(-1).expand(-1, -1, vocab_size)) # NxbxV
        assert (_tmp_beam_logprobs == beam_logprobs).all()
        # beam_seq_logprobs[:,:,t,:] = beam_logprobs.reshape(batch_size, -1, vocab_size)
        beam_seq_logprobs = torch.cat([
            beam_seq_logprobs,
            beam_logprobs.reshape(batch_size, -1, 1, vocab_size)], 2)
        
        new_state = [None for _ in state]
        for _ix in range(len(new_state)):
        #  copy over state in previous beam q to new beam at vix
            new_state[_ix] = state[_ix][state_ix]
        state = new_state

        # selected_ix `(batch_size, beam_size)` -> if extend tree
        ifextend = ( (selected_ix != self.vocab_size) ).unsqueeze(dim=2).repeat(1,1,3)
        src = selected_ix.new_ones(batch_size, beam_size, 3, dtype=torch.long) * ifextend * t
        scatter_index = counter.unsqueeze(dim=2).repeat(1, 1, 3) + torch.arange(0, 3, dtype=torch.long, device=selected_ix.device)
        scatter_index = torch.min(scatter_index, scatter_index.new_ones(scatter_index.size()) * (self.max_seqtree_length - 1))

        beam_parent_idx.scatter_(dim=2, index=scatter_index, src=src)
        all_finished = all_finished | (counter == t) 
        seqLen = seqLen * all_finished + counter * ~all_finished

        counter.add_((selected_ix != self.vocab_size) * 3)

        return beam_seq, beam_parent_idx, beam_seq_logprobs, beam_logprobs_sum, state, counter, seqLen, all_finished


    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        max_seqtree_length = opt.get('max_seqtree_length', 40)
        temperature = opt.get('temperature', 1)
        
        batch_size = init_logprobs.size(0)
        device = init_logprobs.device

        beam_seq_table = torch.LongTensor(batch_size, beam_size, 0).to(device)
        beam_parent_idx_table = torch.LongTensor(batch_size, beam_size, max_seqtree_length).to(device)
        beam_parent_idx_table.fill_(0)
        beam_hidden_states_table = torch.FloatTensor(batch_size*beam_size, max_seqtree_length, self.rnn_size).to(device)
        beam_cell_states_table = torch.FloatTensor(batch_size*beam_size, max_seqtree_length, self.rnn_size).to(device)
        
        # init state
        # init_state `(batch_size, rnn_size)` -> `(batch_size*beam_size, rnn_size)`
        beam_hidden_states_table[:,0,:] = init_state[0].unsqueeze(dim=1).repeat(1,beam_size,1).view(batch_size*beam_size,-1)
        beam_cell_states_table[:,0,:] = init_state[1].unsqueeze(dim=1).repeat(1,beam_size,1).view(batch_size*beam_size,-1)

        beam_seq_logprobs_table = torch.FloatTensor(batch_size, beam_size, 0, self.vocab_size + 1).to(device)
        beam_logprobs_sum_table = torch.zeros(batch_size, beam_size).to(device)
        logprobs = init_logprobs

        # generation finished utils
        counter_table = torch.LongTensor(batch_size, beam_size).to(device)
        counter_table.fill_(1)
        seqLen_table = torch.LongTensor(batch_size, beam_size).to(device)
        seqLen_table.fill_(0)
        all_finished_table = torch.BoolTensor(batch_size, beam_size).to(device)
        all_finished_table.fill_(0)

        done_beams_table = [[] for _ in range(batch_size)]

        for i in range(1, max_seqtree_length):
            beam_seq_table, \
            beam_parent_idx_table, \
            beam_seq_logprobs_table, \
            beam_logprobs_sum_table, \
            (beam_hidden_states_table, \
            beam_cell_states_table), \
            counter_table, \
            seqLen_table, \
            all_finished_table = self.beam_step(logprobs,  
                                            beam_size, 
                                            i-1, 
                                            beam_seq_table,
                                            beam_parent_idx_table,
                                            beam_seq_logprobs_table, 
                                            beam_logprobs_sum_table,
                                            (beam_hidden_states_table,
                                            beam_cell_states_table),
                                            counter_table, 
                                            seqLen_table,
                                            all_finished_table)

            for b in range(batch_size):
                is_end = all_finished_table[b,:]
                if i == max_seqtree_length - 1:
                    is_end.fill_(1)
                for vix in range(beam_size):
                    if is_end[vix]:
                        final_beam = {
                            'seq': beam_seq_table[b,vix].clone(),
                            'seq_idx': beam_parent_idx_table[b,vix].clone(),
                            'seqLen': seqLen_table[b,vix].clone(),
                            'logps': beam_seq_logprobs_table[b,vix].clone(),
                            'unaug_p': beam_seq_logprobs_table[b, vix].sum().item(),
                            'p': beam_logprobs_sum_table[b, vix].item()
                        }
                        done_beams_table[b].append(final_beam)
                beam_logprobs_sum_table[b,is_end] -= 1000
            
            # move the current group one step forward in time
            seqtree = beam_seq_table.view(batch_size*beam_size, -1)
            parent_idx = beam_parent_idx_table.view(batch_size*beam_size, max_seqtree_length)
            p_it = torch.gather(seqtree, dim=1, index=parent_idx[:,i].clone().unsqueeze(1))
            p_it = p_it.squeeze(dim=1)
            p_xt = self.embed(p_it)
            
            hidden_states = beam_hidden_states_table.view(batch_size*beam_size, max_seqtree_length, self.rnn_size)
            cell_states = beam_cell_states_table.view(batch_size*beam_size, max_seqtree_length, self.rnn_size)
            
            p_idx = parent_idx[:,i].clone()
            p_idx = p_idx.unsqueeze(1).unsqueeze(1).expand(batch_size*beam_size, 1, self.hidden_size)
            p_hidden_state = torch.gather(hidden_states, dim=1, index=p_idx).squeeze(dim=1)
            p_cell_state = torch.gather(cell_states, dim=1, index=p_idx).squeeze(dim=1)

            p_state = p_hidden_state, p_cell_state

            if i % 3 == 1:
                s_xt = self.init_input(batch_size*beam_size)
                s_state = self.init_hidden(batch_size*beam_size)
            else:
                s_it = seqtree[:,i-1].clone()
                s_xt = self.embed(s_it)
                s_hidden_state = hidden_states[:,i-1].clone()
                s_cell_state = cell_states[:,i-1].clone()
                s_state = s_hidden_state, s_cell_state
            
            logprobs, _state = self.get_logprobs_state(p_xt, s_xt, p_state, s_state, *args)
            
            # logprobs = logprobs.view(batch_size, beam_size, self.vocab_size+1)
            logprobs = F.log_softmax(logprobs, dim=-1)
            # beam_hidden_states_table[:,:,i,:] = state[0].view(-1, self.rnn_size)
            # beam_cell_states_table[:,:,i,:] = state[1].view(-1, self.rnn_size)
            beam_hidden_states_table[:,i,:] = _state[0]
            beam_cell_states_table[:,i,:] = _state[1]
        
        # all beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[b], key=lambda x: -x['p'])[:beam_size] for b in range(batch_size)]
        # done_beams = [sum(_, []) for _ in done_beams_table]
        return done_beams_table


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size * 2, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res


class MDLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, att_feat_size, bias=True):
        super(MDLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_prob_lm = 0.5
        self.bias = bias

        self.weight_f1 = Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.weight_f2 = Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.weight_H = Parameter(torch.Tensor(2 * hidden_size, 2 * input_size + 2 * hidden_size))
        self.weight_I = Parameter(torch.Tensor(hidden_size, 2 * input_size + 2 * hidden_size + att_feat_size))
        # self.drop_out = nn.Dropout(self.drop_prob_lm)
        if bias:
            self.bias_f1 = Parameter(torch.Tensor(hidden_size))
            self.bias_f2 = Parameter(torch.Tensor(hidden_size))
            self.bias_H = Parameter(torch.Tensor(2 * hidden_size))
            self.bias_I = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_f1', None)
            self.register_parameter('bias_f2', None)
            self.register_parameter('bias_H', None)
            self.register_parameter('bias_I', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2, hx1, hx2, att_feats):
        # self.check_forward_input(input1)
        # self.check_forward_input(input2)
        # self.check_forward_hidden(input1, hx1[0], '[0]')
        # self.check_forward_hidden(input1, hx1[1], '[1]')
        # self.check_forward_hidden(input2, hx2[0], '[0]')
        # self.check_forward_hidden(input2, hx2[1], '[1]')

        H = torch.cat([input1, input2, hx1[0], hx2[0]], dim=1)
        gates = F.linear(H, self.weight_H, self.bias_H)
        ingate, outgate = gates.chunk(2, 1)

        ingate = F.sigmoid(ingate)
        outgate = F.sigmoid(outgate)

        H_hat = torch.cat([input1, input2, hx1[0], hx2[0], att_feats], dim=1)

        cellgate = F.linear(H_hat, self.weight_I, self.bias_I)
        cellgate = torch.tanh(cellgate)

        forgetgate1 = F.linear(torch.cat([input1, hx1[0]], dim=1), self.weight_f1, self.bias_f1)
        forgetgate2 = F.linear(torch.cat([input2, hx2[0]], dim=1), self.weight_f2, self.bias_f2)
        forgetgate1 = F.sigmoid(forgetgate1)
        forgetgate2 = F.sigmoid(forgetgate2)

        next_c = forgetgate1 * hx1[1] + forgetgate2 * hx2[1] + ingate * cellgate
        next_h = outgate * F.tanh(next_c)

        # output = self.drop_out(next_h)
        output = next_h
        return output, (next_h, next_c)


class SpatialAttCore(nn.Module):
    def __init__(self, opt):
        super(SpatialAttCore, self).__init__()
        self.lstm = MDLSTMCell(opt.input_encoding_size, opt.rnn_size, opt.att_hid_size)
        self.attention = Attention(opt)

    def forward(self, p_xt, s_xt, p_state, s_state, fc_feats, att_feats, p_att_feats, att_masks):
        att_state = torch.cat([p_state[0], s_state[0]], dim=1)
        att_res = self.attention(att_state, att_feats, p_att_feats, att_masks)
        output, states = self.lstm(p_xt, s_xt, p_state, s_state, att_res)
        return output, states


class SpatialAttTreeModel(AttTreeModel):
    def __init__(self, opt):
        super(SpatialAttTreeModel, self).__init__(opt)
        self.core = SpatialAttCore(opt)


if __name__ == "__main__":
    data = torch.load('sample_data.pt')
    # fc_feats, att_feats, seqs, att_masks, seqtree, parent_idx, seqtree_mask
    tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['att_masks'],
           data['seqtree'], data['seqtree_idx'], data['seqtree_mask']]
    
    tmp = [_ if _ is None else _.cuda() for _ in tmp]

    fc_feats, att_feats, seqs, att_masks, seqtree, parent_idx, seqtree_mask = tmp
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

    model = SpatialAttTreeModel(opt)
    model = model.cuda()
    (seq, seq_idx, seqLen), seq_logprobs = model(fc_feats, att_feats, att_masks, opt={'sample_method': 'beam_search', 'temperature': 1.0, 'beam_size': 3}, mode='sample')
    # out = model(fc_feats, att_feats, seqs, att_masks, seqtree, parent_idx, seqtree_mask, mode='sample')
    # print(out.size())

    # def crit(input, target, mask):
    #     # truncate to the same size
    #     target = target[:, :input.size(1)]
    #     mask =  mask[:, :input.size(1)]

    #     output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
    #     # Average over each token
    #     output = torch.sum(output) / torch.sum(mask)

    #     return output
    
    # loss = crit(out, seqtree, seqtree_mask)
    # # print(seqtree)
    # # print(seqtree_mask)
    # print(loss)

    # loss.backward()
