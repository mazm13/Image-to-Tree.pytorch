"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.py

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
import skimage.io
from PIL import Image


def encode_captions(imgs, params, wtoi):
    """ 
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed 
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs) # total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='int32') # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='int32')
    label_length = np.zeros(M, dtype='int32')
    caption_counter = 0
    counter = 1
    for i,img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='int32')
        for j,s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
            caption_counter += 1
            for k,w in enumerate(s):
                if k < max_length:
                    Li[j,k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1
        
        counter += n
    
    L = np.concatenate(label_arrays, axis=0) # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length

def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']

    seed(123) # make reproducible
    
    # load the vocab built when prepared tree labels
    tree_imgs = json.load(open(params['input_tree_json'], 'r'))
    itow = tree_imgs['ix_to_word']
    wtoi = {v:k for k,v in itow.items()}
    
    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if w in wtoi else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

    # create output h5 file
    N = len(imgs)
    f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
    f_lb.create_dataset("labels", dtype='int32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='int32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='int32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='int32', data=label_length)
    f_lb.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['images'] = []
    for i,img in enumerate(imgs):
        
        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img: jimg['file_path'] = os.path.join(img.get('filepath', ''), img['filename']) # copy it over, might need
        if 'cocoid' in img:
            jimg['id'] = img['cocoid'] # copy over & mantain an id, if present (e.g. coco ids, useful)
        elif 'imgid' in img:
            jimg['id'] = img['imgid']

        if params['images_root'] != '':
            with Image.open(os.path.join(params['images_root'], img['filepath'], img['filename'])) as _img:
                jimg['width'], jimg['height'] = _img.size

        out['images'].append(jimg)
    
    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--input_tree_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='data.json', help='output json file')
    parser.add_argument('--output_h5', default='data', help='output h5 file')
    parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')

    # options
    parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
