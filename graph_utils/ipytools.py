from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import json
from matplotlib import pyplot as plt
from PIL import Image

ROOT_PATH = '/media/disk/mazm13/dataset/images'
SEQ_PER_IMAGE = 5
INPUT_TREE_JSON = "/home/mazm17/image_captioning/diplomaProj/self-critical.pytorch/data/cocotree.json"
INPUT_JSON = "/home/mazm17/image_captioning/diplomaProj/self-critical.pytorch/data/cocotalk.json"

print("Initialize tree infos from\n{}".format(INPUT_TREE_JSON))
with open(INPUT_JSON) as f:
    INFOS = json.load(f)

print("Initialize talk infos from\n{}".format(INPUT_JSON))
with open(INPUT_JSON) as f:
    INFOS_TALK = json.load(f)

print("Mapping image coco id to its file path")
id_to_fp = {}
for t in INFOS_TALK['images']:
    id_to_fp[t['id']] = t['file_path']


def show_image(file_path):
    image = Image.open(os.path.join(ROOT_PATH, file_path))
    plt.imshow(image)


def display_data(data, vocab=None):

    vocab = vocab or INFOS['ix_to_word']

    batch_size = len(data['infos'])
    bi = random.randint(0, batch_size - 1)
    si = random.randint(0, SEQ_PER_IMAGE - 1)
    
    labels = data['labels'][bi*SEQ_PER_IMAGE+si].tolist()
    seqtree = data['seqtree'][bi*SEQ_PER_IMAGE+si].tolist()
    seqtree_idx = data['seqtree_idx'][bi*SEQ_PER_IMAGE+si].tolist()

    labels_str = " ".join([vocab[str(_)] if _ > 0 else '0' for _ in labels])
    seqtree_str = " ".join([vocab[str(_)] if _ > 0 else '0' for _ in seqtree])
    print("labels: {}".format(labels_str))
    print("seqtree: {}".format(seqtree_str))
    print("seqtree_idx: {}".format(" ".join(list(map(str, seqtree_idx)))))

    show_image( id_to_fp[ data['infos'][bi]['id'] ] )

    return seqtree, seqtree_idx
