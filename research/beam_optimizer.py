import numpy as np

from research.base_optimizer import BaseOptimizer


class BeamOptimizer(BaseOptimizer):

    def __init__(self, english_text, n_beams=10, temperature=1.0):
        super().__init__(english_text, temperature=temperature)
        self.max_steps = 50
        self.n_beams = n_beams
        self.vocab.itos = np.array(self.vocab.itos)
        self.EOS = self.vocab.stoi['</s>']

    def optimize(self):
        n_beams_left = self.n_beams
        finished_beams = []
        finished_beams_probs = np.array([])
        replicated_english_tok_seq = [self.english_tok_seq] * n_beams_left
        init_probs, init_idxs = self.scorer.next_word_probabilities([self.english_tok_seq], [[]])[0].topk(n_beams_left)
        topk_probs = init_probs.unsqueeze(1)
        topk_states = self.vocab.itos[init_idxs.detach().numpy()][:, np.newaxis]
        for _ in range(self.max_steps):
            next_state_probs = self.scorer.next_word_probabilities(replicated_english_tok_seq, topk_states)
            beam_probs = next_state_probs + topk_probs
            probs_flattened = beam_probs.view(-1)
            flat_probs, flat_idxs = probs_flattened.topk(n_beams_left)
            flat_idxs = flat_idxs.detach().numpy()
            beam_idxs = flat_idxs // self.vocab_size
            word_idxs = flat_idxs % self.vocab_size
            topk_probs = flat_probs.unsqueeze(1)
            topk_states = np.hstack((topk_states[beam_idxs], self.vocab.itos[word_idxs][:, np.newaxis]))

            finished_beams_idx = np.flatnonzero(word_idxs == self.EOS)
            n_finished_now = len(finished_beams_idx)
            if n_finished_now > 0:
                finished_beams += topk_states[finished_beams_idx].reshape(n_finished_now, -1).tolist()
                finished_beams_probs = np.concatenate((finished_beams_probs, topk_probs[finished_beams_idx].view(-1).detach().numpy()))
                not_finished_beams_idx = np.flatnonzero(word_idxs != self.EOS)
                topk_probs = topk_probs[not_finished_beams_idx]
                topk_states = topk_states[not_finished_beams_idx]
                n_beams_left -= n_finished_now

            if n_beams_left == 0:
                break

        perm = np.argsort(-finished_beams_probs)
        return [finished_beams[i] for i in perm], finished_beams_probs[perm]