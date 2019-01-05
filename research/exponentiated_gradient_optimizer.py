import numpy as np
import torch
from torch.autograd import Variable

from research.base_optimizer import BaseOptimizer


class ExponentiatedGradientOptimizer(BaseOptimizer):

    def __init__(self, english_text):
        super().__init__(english_text)
        self.max_steps = 1000
        self.vocab.itos = np.array(self.vocab.itos)
        self.EOS = self.vocab.stoi['</s>']
        self.gamma = 0.3

    def optimize(self, init=None, method='max', lr=100, verbose=False, with_score=False):
        L = len(self.english_tok_seq)
        # źle się inicjalizuje - muszę to znormalizować z zachowaniem maksimum
        if init is not None:
            Y = Variable(init / torch.sum(init, 0).unsqueeze(0), requires_grad=True)
        else:
            Y = Variable(torch.ones(self.vocab_size, 2*L), requires_grad=True)

        for t in range(10):
            score = -self.scorer.score_tokenized_texts([self.english_tok_seq], [Y], relaxed=True, method=method)
            score.backward()

            if verbose:
                print('Score at step ', t, "=", score, 'max grad component =', Y.grad.max() if Y.grad is not None else '', 'lr = ', lr)

            with torch.no_grad():
                Y *= torch.exp(-lr * Y.grad) #/ torch.sum(torch.exp(lr*Y.grad), 0).unsqueeze(0)
                Y /= torch.sum(Y, 0).unsqueeze(0)
                Y.grad.zero_()


        I = torch.max(Y, 0)
        if verbose:
            print('Max(Y) = ', I[0])
        if with_score:
            return list([self.vocab.itos[i] for i in I[1]]), score

        return list([self.vocab.itos[i] for i in I[1]])