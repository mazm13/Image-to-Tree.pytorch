from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import json
import argparse

import numpy as np
import h5py

from graph_utils import graph


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                w = w.lower()
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr]
    vocab = [w for w,n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len+1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    def __gloss(x):
        if x == "ROOT":
            return x
        else:
            x = x.lower()
            if counts.get(x, 0) > count_thr:
                return x
            else:
                return 'UNK'
    
    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt, depends = sent['tokens'], sent['depends']
            caption = [w.lower() if counts.get(w.lower(),0) > count_thr else 'UNK' for w in txt]
            for d in depends:
                d['dependentGloss'] = __gloss(d['dependentGloss'])
                d['governorGloss'] = __gloss(d['governorGloss'])

            img['final_captions'].append({'caption': caption, 'depends': depends})

    vocab.append('ROOT')
    vocab.append('EOB')

    return vocab


def encode_captions(imgs, params, wtoi):
    max_length = params['max_length']
    max_treearray_length = params['max_treearray_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='int32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='int32')
    label_length = np.zeros(M, dtype='int32')

    tree_arrays = []
    tree_array_idx = []
    tree_array_length = np.zeros(M, dtype='int32')

    caption_counter = 0
    counter = 1
    for i,img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='int32')
        Ti = np.zeros((n, max_treearray_length), dtype='int32')
        Fi = np.zeros((n, max_treearray_length), dtype='int32')

        for j,fs in enumerate(img['final_captions']):
            s = fs['caption']
            label_length[caption_counter] = min(max_length, len(s))
            for k,w in enumerate(s):
                if k < max_length:
                    Li[j,k] = wtoi[w]
            
            tarray, tarray_idx = graph.depends2array(fs['depends'])
            tree_array_length[caption_counter] = min(max_treearray_length, len(tarray))
            caption_counter += 1

            for k,w in enumerate(tarray):
                if k < max_treearray_length:
                    Ti[j,k] = wtoi[w]
            for k,w in enumerate(tarray_idx):
                if k < max_treearray_length:
                    Fi[j,k] = w
        
        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        tree_arrays.append(Ti)
        tree_array_idx.append(Fi)

        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1
        
        counter += n
        print("Writing to h5 file: {:.2f}%".format(i * 100 / N), end="\r")

    print()

    L = np.concatenate(label_arrays, axis=0) # put all the labels together
    print(L.shape)
    print(M)
    assert L.shape[0] == M, 'L lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    T = np.concatenate(tree_arrays, axis=0)  # put all treearrays together
    F = np.concatenate(tree_array_idx, axis=0)
    assert T.shape[0] == M, 'T lengths don\'t match? that\'s weird'
    assert F.shape[0] == M, 'F lengths don\'t match? that\'s weird'
    assert np.all(tree_array_length > 0), 'error: some caption had no trees?'

    print('encoded captions to array of size ', L.shape)
    return L, T, F, label_start_ix, label_end_ix, label_length, tree_array_length


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    
    # create the vocab
    vocab = build_vocab(imgs, params)
    # print(json.dumps(imgs[0], indent=2))
    itow = {i+1:w for i,w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)}  # inverse table

    print(len(vocab))

    # encode captions in large arrays, ready to ship to hdf5 file
    L, T, F, label_start_ix, label_end_ix, label_length, tree_array_length = encode_captions(imgs, params, wtoi)
    
    # create output h5 file
    N = len(imgs)
    f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
    f_lb.create_dataset("labels", dtype='int32', data=L)
    f_lb.create_dataset("treearray", dtype='int32', data=T)
    f_lb.create_dataset("treearray_idx", dtype='int32', data=F)
    f_lb.create_dataset("label_start_ix", dtype='int32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='int32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='int32', data=label_length)
    f_lb.create_dataset("treearray_length", dtype='int32', data=tree_array_length)
    f_lb.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['images'] = []
    for i,img in enumerate(imgs):
        jimg = {}
        jimg['split'] = 'train'
        jimg['id'] = img['img_id']
        if 'filename' in img:
            jimg['file_path'] = os.path.join(img.get('filepath', ''), img['filename'])
        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input_json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_h5', default='dataset/cocotree', help='output h5 file')
    parser.add_argument('--output_json', default='dataset/cocotree.json', help='output json file')

    # options
    parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--max_treearray_length', default=40, type=int, help='max length of the vector representing a tree')
    
    args = parser.parse_args()
    params = vars(args)
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    start_time = time.time()
    main(params)
    print("Finish, takes {} seconds.".format(time.time() - start_time))
