import math
import random
import string

from research.scorer import Scorer
from research.utils import load_vocabulary, Tokenizer


class SimulatedAnnealingOptimizer(object):

    def __init__(self, english_text):
        self.tokenize_english_text(english_text)
        self.specials = ['<blank>', '<unk>', 0, '<s>', '</s>']
        self.scorer = Scorer()
        self.energy = lambda x: self.scorer.score_tokenized_texts([self.english_tok_seq for _ in range(len(x))], x)
        self.temperature = lambda x: math.exp(-x)
        self.k_max = 10000
        self.possible_chars = [ch for ch in string.ascii_letters] + [' ']
        self.vocab = load_vocabulary()

    def tokenize_english_text(self, english_text):
        tokenizer = Tokenizer()
        self.english_tok_seq = tokenizer.tokenize(english_text)

    def optimize(self):
        state = [random.choice(self.vocab.itos)]
        print('initial state = ', state)
        old_energy = self.energy(state)
        for k in range(self.k_max):
            T = self.temperature(float(k) / self.k_max)
            neighbor_states = [state + [next_token] for next_token in self.vocab.itos if next_token not in self.specials] + \
                              [state[:-1] + [next_token] for next_token in self.vocab.itos if next_token not in self.specials]

            minibatch = [random.choice(neighbor_states) for _ in range(1000)]
            curr_energies = np.array(self.energy(minibatch))
            imax = curr_energies.argmax()
            neighbor_state = minibatch[imax]
            curr_energy = curr_energies[imax]

            delta_energy = curr_energy - old_energy
            if delta_energy > 0:
                state = neighbor_state
                old_energy = curr_energy
            elif random.uniform(0, 1) <= math.exp(delta_energy / T):
                state = neighbor_state
                old_energy = curr_energy

            if k % 50 == 0:
                print('k = ', k, 'state = ', state, ' energy = ', old_energy, ' temperature = ', T)
        return state