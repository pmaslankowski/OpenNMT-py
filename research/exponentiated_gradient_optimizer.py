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

    def optimize(self, init=None, method='multiplication', start_lr=1.0, verbose=False, with_score=False):
        L = len(self.english_tok_seq)
        if init is not None:
            Y = Variable(torch.exp(init), requires_grad=True)
        else:
            Y = Variable(torch.ones(self.vocab_size, 2*L), requires_grad=True)

        lr = start_lr
        prev_score = 1000000.
        for t in range(10):
            if lr < 10e-6:
                break

            score = -self.scorer.score_tokenized_texts([self.english_tok_seq], [Y], relaxed=True, method=method)

            compute_grad = True
            if score < prev_score:
                lr = start_lr
            else:
                score = prev_score
                Y = prev_Y.clone()
                Y.grad = prev_grad.clone()
                lr *= 0.5
                compute_grad = False

            if verbose:
                print('Score at step ', t, "=", score, 'max grad component =', Y.grad.max() if Y.grad is not None else '', 'lr = ', lr)
                print('\t cscore = ', -self.scorer.score_tokenized_texts([self.english_tok_seq], [Y], relaxed=True, method=method, normalize=True))
                I = torch.max(Y, 0)
                translation = list([self.vocab.itos[i] for i in I[1]])
                print('\tdscore = ', -self.scorer.score_tokenized_texts([self.english_tok_seq], [translation], method=method, normalize=True))
                print(' '.join(translation))
                print()

            if compute_grad:
                score.backward()
                prev_Y = Y.clone()
                prev_grad = Y.grad.clone()
                prev_score = score

            with torch.no_grad():
                if compute_grad:
                    Y *= torch.exp(-lr * Y.grad)
                    Y /= torch.sum(Y, 0).unsqueeze(0)
                    Y.grad.zero_()
                else:
                    Y *= torch.exp(-lr * prev_grad)
                    Y /= torch.sum(Y, 0).unsqueeze(0)

        I = torch.max(Y, 0)
        if verbose:
            print('Max(Y) = ', I[0])
        if with_score:
            return list([self.vocab.itos[i] for i in I[1]]), score

        return list([self.vocab.itos[i] for i in I[1]])