import argparse
import onmt
import onmt.opts as opts
import sentencepiece as spm
import torch
from copy import deepcopy
from research import consts


def load_vocabulary():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, _, _ = \
        onmt.model_builder.load_test_model(consts.OPT, dummy_opt.__dict__)

    return fields['tgt'].vocab


class Tokenizer(object):

    def __init__(self):
        self.sentencepiece_tokenizer = spm.SentencePieceProcessor()
        self.sentencepiece_tokenizer.Load(consts.SENTENCEPIECE_MODEL_PATH)

    def tokenize(self, text):
        return self.sentencepiece_tokenizer.EncodeAsPieces(text)


class Aligner(object):

    def __init__(self, debug = False):
        self.add_cost = 1
        self.subst_cost = 1
        self.del_cost = 1
        self.debug = debug

    def align(self, sen1, sen2):
        n, m = len(sen1), len(sen2)
        dp = [[-1 for _ in range(m+1)] for _ in range(n+1)]
        pred = {}
        for i in range(n+1):
            dp[i][0] = self.del_cost * i
        for i in range(m+1):
            dp[0][i] = self.del_cost * i

        for i in range(1, n+1):
            for j in range(1, m+1):
                if sen1[i-1] == sen2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    pred[(i, j)] = (i-1, j-1, 'NOP')
                else:
                    add = dp[i][j-1] + self.add_cost, i, j-1, f'ADD({sen2[j-1]})'
                    subst = dp[i - 1][j - 1] + self.subst_cost, i-1, j-1, f'SUBST({sen1[i-1]} -> {sen2[j-1]})'
                    delete = dp[i-1][j] + self.del_cost, i-1, j, f'DEL({sen1[i-1]})'

                    # we select state with minimum cost
                    cost, pi, pj, op = min([add, subst, delete], key=lambda x: x[0])
                    dp[i][j] = cost
                    pred[(i, j)] = (pi, pj, op)

        ops = []
        current = (n, m)
        while current in pred:
            i, j, op = pred[current]
            ops += [op]
            current = i, j
        ops.reverse()

        res1 = deepcopy(sen1)
        res2 = deepcopy(sen2)
        PLACEHOLDER = '[PLACEHOLDER]'
        for i, op in enumerate(ops):
            if op.startswith('ADD'):
                res1.insert(i, PLACEHOLDER)
            elif op.startswith('DEL'):
                res2.insert(i, PLACEHOLDER)

        if self.debug:
            print(f'Allignment cost = {dp[n-1][m-1]}')
            print(f'Ops: ', ' '.join(ops))

        return res1, res2


class OneHotEncoder(object):

    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, X):
        length, batch_size, code_size = X.shape
        res = torch.FloatTensor(length, batch_size, 1, len(self.vocab.itos))
        res.zero_()
        res.scatter_(3, X.unsqueeze(2), 1)
        return res

