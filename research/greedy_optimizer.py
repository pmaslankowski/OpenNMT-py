from research.base_optimizer import BaseOptimizer


class GreedyOptimizer(BaseOptimizer):

    def __init__(self, english_text):
        super().__init__(english_text)
        self.max_steps = 1000

    def optimize(self):
        state = []
        for _ in range(self.max_steps):
            next_state_probs = self.scorer.next_word_probabilities([self.english_tok_seq], [state])[0]
            _, idx = next_state_probs.topk(1)
            next_word = self.vocab.itos[idx.view(-1)]
            state += [next_word]
            if next_word == '</s>':
                break

        return state