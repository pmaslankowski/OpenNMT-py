import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from research.base_optimizer import BaseOptimizer


class ContinuousOptimizer(BaseOptimizer):

    def __init__(self, english_text, temperature=1.0):
        super().__init__(english_text, temperature=temperature)
        self.max_steps = 1000
        self.vocab.itos = np.array(self.vocab.itos)
        self.EOS = self.vocab.stoi['</s>']
        self.gamma = 0.3

    def optimize(self, init=None, method='max', verbose=False, with_score=False, start_lr=100, max_iters = 100):
        L = len(self.english_tok_seq)
        if init is not None:
            R = Variable(init, requires_grad=True)
        else:
            R = Variable(torch.ones(self.vocab_size, 2*L), requires_grad=True)

        lr = start_lr
        prev_score = 1000000.
        for t in range(max_iters):
            if lr < 10e-6:
                break

            Y = F.softmax(R, 0)
            score = -self.scorer.score_tokenized_texts([self.english_tok_seq], [Y], relaxed=True, method=method, normalize=True)
            compute_grad = True
            if score < prev_score:
                lr = start_lr
            else:
                score = prev_score
                R = prev_R.clone()
                R.grad = prev_grad.clone()
                lr *= 0.5
                compute_grad = False

            if verbose:
                Y_for_logging = F.softmax(R, 0)
                print('Step', t, ', loss score = ', score, 'max grad component =', R.grad.max().item() if R.grad is not None else '', 'lr = ', lr)
                print('\t cscore = ', -self.scorer.score_tokenized_texts([self.english_tok_seq], [Y_for_logging], relaxed=True, method=method, normalize=True))
                I = torch.max(Y_for_logging, 0)
                translation = list([self.vocab.itos[i] for i in I[1]])
                print('\tdscore = ', -self.scorer.score_tokenized_texts([self.english_tok_seq], [translation], method=method, normalize=True))
                print(' '.join(translation))
                print()

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
        translation = list([self.vocab.itos[i] for i in I[1]])
        if verbose:
            print('Max(Y) = ', I[0])
            print('Final dscore', -self.scorer.score_tokenized_texts([self.english_tok_seq], [translation], relaxed=False, method=method))
        if with_score:
            return translation, score

        return translation

    def optimize_with_trace(self, init=None, method='max', start_lr=100, max_iters = 100):
        L = len(self.english_tok_seq)
        if init is not None:
            R = Variable(init, requires_grad=True)
        else:
            R = Variable(torch.ones(self.vocab_size, 2*L), requires_grad=True)

        lr = start_lr
        prev_score = 1000000.
        res = []
        for t in range(max_iters):
            if lr < 10e-6:
                break
            Y = F.softmax(R, 0)
            score = -self.scorer.score_tokenized_texts([self.english_tok_seq], [Y], relaxed=True, method=method, normalize=True)
            compute_grad = True
            if score < prev_score:
                lr = start_lr
            else:
                score = prev_score
                R = prev_R.clone()
                R.grad = prev_grad.clone()
                Y = F.softmax(R, 0)
                lr *= 0.5
                compute_grad = False

            I = torch.max(Y, 0)
            translation = list([self.vocab.itos[i] for i in I[1]])
            res.append({'step': t, 'loss_score': score,
                        'cscore': -self.scorer.score_tokenized_texts([self.english_tok_seq], [Y], relaxed=True, method=method, normalize=True),
                        'translation': translation, 'dscore': -self.scorer.score_tokenized_texts([self.english_tok_seq], [translation], method=method, normalize=True)})


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

        return res