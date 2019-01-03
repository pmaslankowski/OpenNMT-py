import argparse
import re
import subprocess
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

BLEU_SCRIPT_PATH = '/home/pma/Dropbox/Documents/Studia/Semestr_7/Praca_Inzynierska/OpenNMT-py/tools/multi-bleu.perl'
def bleu(translation, reference):
    with open('tmp.de', 'w+') as f:
        f.write(reference)

    process = subprocess.Popen(['perl', BLEU_SCRIPT_PATH, 'tmp.de'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    process.stdin.write(translation.encode('utf-8'))
    process.stdin.close()

    output = process.stdout.read().decode('utf-8')
    res = re.match('^BLEU = (.*?),', output).group(1)
    return float(res)


class Tokenizer(object):

    def __init__(self):
        self.sentencepiece_tokenizer = spm.SentencePieceProcessor()
        self.sentencepiece_tokenizer.Load(consts.SENTENCEPIECE_MODEL_PATH)

    def tokenize(self, text):
        return self.sentencepiece_tokenizer.EncodeAsPieces(text)

    def detokenize(self, tokens):
        return self.sentencepiece_tokenizer.DecodePieces(tokens)


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

    def encode(self, X, v=1.):
        length, batch_size, code_size = X.shape
        res = torch.FloatTensor(length, batch_size, 1, len(self.vocab.itos))
        res.zero_()
        res.scatter_(3, X.unsqueeze(2), v)
        return res


class RelaxedTargetField(object):

    def __init__(self):
        self.is_target = True
        vocab_size = 31538
        tok_begin_idx = 2
        tok_end_idx = 3
        self.tok_begin_vec = torch.zeros(vocab_size)
        self.tok_begin_vec[tok_begin_idx] = 1.
        self.tok_end_vec = torch.zeros(vocab_size)
        self.tok_end_vec[tok_end_idx] = 1.

    def process(self, batch, device=None):
        return torch.stack([torch.stack(b, 0) for b in batch], 1)

    def preprocess(self, x):
        return (self.tok_begin_vec,) + x + (self.tok_end_vec,)

