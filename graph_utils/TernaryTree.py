from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from queue import Queue

from .graph import TernaryTree as SuperTernaryTree


def _clone(x):
    if isinstance(x, tuple):
        return tuple(_.clone() for _ in x)
    else:
        return x.clone()


class TernaryVectex(object):
    def __init__(self, idx, word, state=None):
        self.idx = idx
        self.word = word
        self.state = state
        self.logprob = 0

        # children
        self.a = None
        self.b = None
        self.c = None

    @classmethod
    def from_vectex(cls, vectex):
        return cls(vectex.idx, vectex.word)
    
    def clone(self):
        v = TernaryVectex(self.idx, self.word.clone(), _clone(self.state))
        v.logprob = self.logprob
        return v


class TernaryTree(object):
    def __init__(self):
        self.vectices = {}
        self.root = None
        self.size = 0
    
        self.logprob = 0

    def clone(self):
        get_idx = lambda x: -1 if x is None else x.idx
        adjacency = {}
        for v in self.vectices.values():
            adjacency.update({v.idx: (get_idx(v.a), get_idx(v.b), get_idx(v.c))})

        t = TernaryTree()
        t.vectices = {k: v.clone() for k,v in self.vectices.items()}
        t.root = t.vectices[0] if len(t.vectices) > 0 else None
        for v in t.vectices.values():
            if adjacency[v.idx][0] == -1:
                continue
            v.a = t.vectices[adjacency[v.idx][0]]
            v.b = t.vectices[adjacency[v.idx][1]]
            v.c = t.vectices[adjacency[v.idx][2]]
        t.logprob = self.logprob
        return t
    
    def __preorder_traversal(self, vectex, words):
        if vectex.a is not None and vectex.a.word != "EOB":
            self.__preorder_traversal(vectex.a, words)
        words.append(vectex.word)
        if vectex.b is not None and vectex.b.word != "EOB":
            self.__preorder_traversal(vectex.b, words)
        if vectex.c is not None and vectex.c.word != "EOB":
            self.__preorder_traversal(vectex.c, words)

    def decode(self, vocab):
        for v in self.vectices.values():
            v.word = vocab[str(v.word.item())]
        words = []
        self.__preorder_traversal(self.root, words)
        return " ".join(words[1:])
    
    def pprint(self):
        for v in self.vectices.values():
            print(str(v.idx) + ': ' + v.word)
            if v.a is not None:
                print(v.a.word + '_' + v.b.word + '_' + v.c.word)
            print()
