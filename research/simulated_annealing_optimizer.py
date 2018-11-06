import math
import random

from research.base_optimizer import BaseOptimizer


class SimulatedAnnealingOptimizer(BaseOptimizer):

    def __init__(self, english_text):
        super().__init__(english_text)
        self.k_max = 10000

    def temperature(self, x):
        return math.exp(-x)

    def optimize(self):
        state = []
        print('initial state = ', state)
        curr_energy = float('-inf')
        for k in range(self.k_max):
            T = self.temperature(float(k) / self.k_max)
            change_last_word = random.uniform(0, 1) < 0.5
            last_word = None
            if state and change_last_word:
                last_word = state[-1]
                state = state[:-1]

            next_state_probs = self.scorer.next_word_probabilities([self.english_tok_seq], [state])[0]
            idx = random.randint(0, self.vocab_size)
            next_energy = next_state_probs[idx]
            delta_energy = next_energy - curr_energy
            if delta_energy > 0:
                state += [self.vocab.itos[idx]]
                curr_energy = next_energy
            elif random.uniform(0, 1) <= math.exp(delta_energy / T):
                state += [self.vocab.itos[idx]]
                curr_energy = next_energy
            elif state and change_last_word:
                state += [last_word]

            if k % 50 == 0:
                print('k = ', k, 'state = ', state, ' energy = ', curr_energy, ' temperature = ', T)

        return state