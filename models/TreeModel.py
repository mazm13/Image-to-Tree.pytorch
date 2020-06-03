from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import misc.utils as utils


class TreeModel(nn.Module):
    def __init__(self):
        super(TreeModel, self).__init__()
    
    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_'+mode)(*args, **kwargs)
    
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
            counter = counter.gather(1, beam_ix)
            seqLen = seqLen.gather(1, beam_ix)
            all_finished = all_finished.gather(1, beam_ix)

        # beam_seq[:,:,t] = selected_ix
        # TODO: why not assign with above may?
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
        ifextend = ( (selected_ix != self.vocab_size) & (selected_ix != 0) ).unsqueeze(dim=2).repeat(1,1,3)
        src = selected_ix.new_ones(batch_size, beam_size, 3, dtype=torch.long) * ifextend * t
        scatter_index = counter.unsqueeze(dim=2).repeat(1, 1, 3) + torch.arange(0, 3, dtype=torch.long, device=selected_ix.device)
        scatter_index = torch.min(scatter_index, scatter_index.new_ones(scatter_index.size()) * (self.max_seqtree_length - 1))

        beam_parent_idx.scatter_(dim=2, index=scatter_index, src=src)
        all_finished = all_finished | (counter == t) 
        seqLen = seqLen * all_finished + counter * ~all_finished
        # print(counter, t, all_finished, seqLen)

        # counter.add_(((selected_ix != self.vocab_size) & (selected_ix != 0) ) * 3)
        counter.add_((selected_ix != self.vocab_size) * 3)

        return beam_seq, beam_parent_idx, beam_seq_logprobs, beam_logprobs_sum, state, counter, seqLen, all_finished

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        max_seqtree_length = opt.get('max_seqtree_length', 40)
        temperature = opt.get('temperature', 1)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        suppress_EOB_factor = opt.get('suppress_EOB_factor', 1)
        # assert suppress_EOB_factor > 1
        
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
            if suppress_EOB_factor > 1:
                logprobs[:,self.vocab_size] = logprobs[:,self.vocab_size] * suppress_EOB_factor
            logprobs[:,0] = logprobs[:,0] - 1000

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
                            'p': beam_logprobs_sum_table[b, vix].item(),
                            'counter': counter_table[b,vix].item()
                        }
                        final_beam['p'] = length_penalty((final_beam['seq'] != self.vocab_size).sum().item(), final_beam['p'])
                        # print(final_beam['seq'].size(), final_beam['seqLen'])
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
        done_beams_table = [sorted(done_beams_table[b], key=lambda x: -x['p']) for b in range(batch_size)]
        # done_beams_table = [sorted(done_beams_table[b], key=lambda x: -x['p'])[:beam_size] for b in range(batch_size)]
        # done_beams = [sum(_, []) for _ in done_beams_table]
        return done_beams_table
