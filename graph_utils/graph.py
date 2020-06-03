from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from queue import Queue


class Vertex(object):
    def __init__(self, idx, word=None):
        # attributes
        self.idx = idx
        self.word = word

        # children
        self.chd = list()


class Graph(object):
    def __init__(self):
        self.vectices = dict()
        self.edges = list()
        self.root = None
    
    def add_vertex(self, idx, word):
        if idx in self.vectices:
            return self.vectices[idx]
        new_vectex = Vertex(idx, word)
        self.vectices.update({idx: new_vectex})
        if idx == 0:
            self.root = new_vectex
        return new_vectex
    
    def add_edge(self, subj_idx, obj_idx):
        if subj_idx not in self.vectices:
            raise ValueError("subj_idx: {} not exists.".format(subj_idx))
        if obj_idx not in self.vectices:
            raise ValueError("obj_idx: {} not exists.".format(obj_idx))

        if obj_idx not in self.vectices[subj_idx].chd:
        # print("{}->{}".format(subj_idx, obj_idx))
        # if obj_idx not in self.vectices[subj_idx].chd:
            self.vectices[subj_idx].chd.append(obj_idx)
    
    def preorder_traversal(self, vectex, words):
        chd = sorted(vectex.chd)
        lc = [_ for _ in chd if _ < vectex.idx]
        rc = [_ for _ in chd if _ > vectex.idx]
        for v in lc:
            self.preorder_traversal(self.vectices[v], words)
        words.append(vectex.word)
        for v in rc:
            self.preorder_traversal(self.vectices[v], words)

    def __str__(self):
        words = []
        self.preorder_traversal(self.root, words)
        return " ".join(words)

    def pprint(self):
        for k, v in self.vectices.items():
            print(k, v.word)
            print(v.chd)

    def __to_ternarytree(self, vectex):
        t_vectex = TernaryVectex.from_vectex(vectex)
        chd = sorted(vectex.chd)
        lc = [_ for _ in chd if _ < vectex.idx]
        rc = [_ for _ in chd if _ > vectex.idx]
        if lc:
            subtree = [self.__to_ternarytree(self.vectices[v]) for v in lc]
            t_vectex.a = subtree[0]
            for i in range(len(subtree) - 1):
                subtree[i].c = subtree[i+1]
        if rc:
            subtree = [self.__to_ternarytree(self.vectices[v]) for v in rc]
            t_vectex.b = subtree[0]
            for i in range(len(subtree) - 1):
                subtree[i].c = subtree[i+1]
        return t_vectex

    def to_ternarytree(self):
        ternarytree = TernaryTree()
        ternarytree.root = self.__to_ternarytree(self.root)
        ternarytree.size = len(self.vectices)
        ternarytree.update_padding()
        return ternarytree


class TernaryVectex(object):
    def __init__(self, idx, word):
        self.idx = idx
        self.word = word

        # children
        self.a = None
        self.b = None
        self.c = None

    @classmethod
    def from_vectex(cls, vectex):
        return cls(vectex.idx, vectex.word)


class TernaryTree(object):
    def __init__(self):
        self.vectices = {}
        self.root = None
        self.size = 0
    
    def __update_padding(self, vectex):
        self.vectices.update({vectex.idx: vectex})
        if vectex.a is None:
            vectex.a = TernaryVectex(self.size, "EOB")
            self.vectices.update({self.size: vectex.a})
            self.size += 1
        else:
            self.__update_padding(vectex.a)
        if vectex.b is None:
            vectex.b = TernaryVectex(self.size, "EOB")
            self.vectices.update({self.size: vectex.b})
            self.size += 1
        else:
            self.__update_padding(vectex.b)
        if vectex.c is None:
            vectex.c = TernaryVectex(self.size, "EOB")
            self.vectices.update({self.size: vectex.b})
            self.size += 1
        else:
            self.__update_padding(vectex.c)
    
    def update_padding(self):
        self.__update_padding(self.root)
    
    def to_array(self):
        queue = Queue()
        queue.put((self.root, 0))

        array = [self.root.word]
        array_idx = [-1]

        while not queue.empty():
            v, v_idx = queue.get()
            counter = len(array)
            array.extend([v.a.word, v.b.word, v.c.word])
            array_idx.extend([v_idx, v_idx, v_idx])
            if v.a.word != "EOB":
                queue.put((v.a, counter + 0))
            if v.b.word != "EOB":
                queue.put((v.b, counter + 1))
            if v.c.word != "EOB":
                queue.put((v.c, counter + 2))
        assert len(array) == len(array_idx)
        return array, array_idx

    def pprint(self):
        for k, v in self.vectices.items():
            print(k, v.word)
            chd = [v.a, v.b, v.c]
            chd = [v.word if v is not None else "None" for v in chd]
            print(", ".join(chd))

    def __preorder_traversal(self, vectex, words):
        if vectex.a is not None and vectex.a.word != "EOB":
            self.__preorder_traversal(vectex.a, words)
        words.append(vectex.word)
        if vectex.b is not None and vectex.b.word != "EOB":
            self.__preorder_traversal(vectex.b, words)
        if vectex.c is not None and vectex.c.word != "EOB":
            self.__preorder_traversal(vectex.c, words)

    def __str__(self):
        words = []
        self.__preorder_traversal(self.root, words)
        return " ".join(words[1:])


def depends2array(depends):
    g = Graph()
    for d in depends:
        subj_idx = d['governor']
        subj_word = d['governorGloss']
        obj_idx = d['dependent']
        obj_word = d['dependentGloss']

        g.add_vertex(subj_idx, subj_word)
        g.add_vertex(obj_idx, obj_word)
        g.add_edge(subj_idx, obj_idx)
    
    ttree = g.to_ternarytree()
    return ttree.to_array()


def decode_array(seq, seq_idx):
    ttree = TernaryTree()
    for i, (w, ix) in enumerate(zip(seq, seq_idx)):
        if w == "<PAD>":
            continue
        tnode = TernaryVectex(i, w)
        ttree.vectices.update({i: tnode})
        if i == 0:
            ttree.root = tnode
        elif i % 3 == 1:
            ttree.vectices[ix].a = ttree.vectices[i]
        elif i % 3 == 2:
            ttree.vectices[ix].b = ttree.vectices[i]
        elif i % 3 == 0:
            ttree.vectices[ix].c = ttree.vectices[i]

    return ttree


if __name__ == '__main__':
    with open('small_dataset.json', 'r') as f:
        dset = json.load(f)
    depends = dset[0]['dependency']

    g = Graph()
    for d in depends:
        print(d)
        subj_idx = d['governor']
        subj_word = d['governorGloss']
        obj_idx = d['dependent']
        obj_word = d['dependentGloss']

        g.add_vertex(subj_idx, subj_word)
        g.add_vertex(obj_idx, obj_word)
        g.add_edge(subj_idx, obj_idx)
    
    print(str(g))

    exit()
    ttree = g.to_ternarytree()

    arr = ttree.to_array()
    print(arr)
    print(str(ttree))
    inv_ttree = decode_array(arr[0], arr[1])
    # inv_ttree.pprint()
    print(str(inv_ttree))
