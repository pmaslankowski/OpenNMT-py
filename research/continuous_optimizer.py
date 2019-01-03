import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from research.base_optimizer import BaseOptimizer


class ContinuousOptimizer(BaseOptimizer):

    def __init__(self, english_text):
        super().__init__(english_text)
        self.max_steps = 1000
        self.vocab.itos = np.array(self.vocab.itos)
        self.EOS = self.vocab.stoi['</s>']
        self.gamma = 0.3

    def optimize(self, init=None, method='max', verbose=False, with_score=False):
        L = len(self.english_tok_seq)
        if init is not None:
            R = Variable(init, requires_grad=True)
        else:
            R = Variable(torch.ones(self.vocab_size, 2*L), requires_grad=True)

        lr = 100
        prev_score = 1000000.
        for t in range(100):
            Y = F.softmax(R, 0)
            score = -self.scorer.score_tokenized_texts([self.english_tok_seq], [Y], relaxed=True, method=method)
            compute_grad = True
            if score < prev_score:
                lr *= 0.99
            else:
                score = prev_score
                R = prev_R
                R.grad = prev_grad
                lr *= 0.6
                compute_grad = False

            if verbose:
                print('Score at step ', t, "=", score, 'max grad component =', R.grad.max() if R.grad is not None else '', 'lr = ', lr)

            if compute_grad:
                score.backward()
                prev_R = R.clone()
                prev_grad = R.grad.clone()
                prev_score = score

            with torch.no_grad():
                if compute_grad:
                    R -= lr * R.grad
                    R.grad.zero_()
                else:
                    R -= lr * prev_grad

        Y = F.softmax(R, 0)
        I = torch.max(Y, 0)
        if verbose:
            print('Max(Y) = ', I[0])
        if with_score:
            return list([self.vocab.itos[i] for i in I[1]]), score

        return list([self.vocab.itos[i] for i in I[1]])