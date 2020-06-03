from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import argparse


def dep2sent(depends):
    tokens = {}
    for it in depends:
        posi, word = it['dependent'], it['dependentGloss']
        tokens[posi] = word
    tokens = [tokens[i+1] for i in range(len(tokens))]
    raw = " ".join(tokens)
    print(raw)
    print(tokens)
    return {"tokens": tokens, "raw": raw, "depends": depends}

def main(params):
    imgs = json.load(open(params['input_json'], 'r'))

    imgs_dict = {}
    for img in imgs:
        sent = dep2sent(img['depends'])
        if img['img_id'] in imgs_dict:
            imgs_dict[img['img_id']].append(sent)
        else:
            imgs_dict[img['img_id']] = [sent]
    
    out = [{'sentences': v, 'img_id': k} for k, v in imgs_dict.items()]
    with open(params['output_json'], 'w') as f:
        json.dump(out, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input_json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='data.json', help='output json file')

    args = parser.parse_args()
    params = vars(args)
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
