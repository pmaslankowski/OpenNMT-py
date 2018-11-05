import sentencepiece as spm

from onmt.translate.translator import build_translator
import random
import string
import math

MODEL_PATH = '../data/transformer/averaged-10-epoch.pt'
SENTENCEPIECE_MODEL_PATH = '../data/transformer/sentencepiece.model'
SOURCE_PATH = 'example.atok'
TARGET_PATH = 'example_target.atok'
OUTPUT_PATH = 'example_output.atok'

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class TranslationScorer(object):

    def __init__(self):
        self.opt = AttrDict({
            'models': [MODEL_PATH], 'data_type': 'text', 'src': SOURCE_PATH, 'src_dir': '',
            'tgt': TARGET_PATH, 'output': OUTPUT_PATH, 'report_bleu': False, 'report_rouge': False,
            'dynamic_dict': False, 'share_vocab': False, 'fast': False, 'beam_size': 5, 'min_length': 0, 'max_length': 100,
            'max_sent_length': None, 'stepwise_penalty': False, 'length_penalty': 'none', 'coverage_penalty': 'none',
            'alpha': 0.0, 'beta': -0.0, 'block_ngram_repeat': 0, 'ignore_when_blocking': [], 'replace_unk': False,
            'verbose': True, 'log_file': '', 'attn_debug': False, 'dump_beam': '', 'n_best': 1, 'batch_size': 30, 'gpu': -1,
            'sample_rate': 16000, 'window_size': 0.02, 'window_stride': 0.01, 'window': 'hamming', 'image_channel_size': 3
        })
        self.sentencepiece_processor = spm.SentencePieceProcessor()
        self.sentencepiece_processor.Load(SENTENCEPIECE_MODEL_PATH)
        self.translator = build_translator(self.opt, report_score=True)

    # returns log likehood of german_translation given english_text
    def score_texts(self, english_text, german_translation):
        tokenized_english_text = self.sentencepiece_processor.EncodeAsPieces(english_text)
        tokenized_german_translation = self.sentencepiece_processor.EncodeAsPieces(german_translation)
        return self.translator.score_target(src_data_iter=[' '.join(tokenized_english_text)],
                                       tgt_data_iter=[' '.join(tokenized_german_translation)],
                                       batch_size=self.opt.batch_size)

    def score(self):
        return self.translator.score_target(src_path=self.opt.src,
                             tgt_path=self.opt.tgt,
                             src_dir=self.opt.src_dir,
                             batch_size=self.opt.batch_size)



class SimulatedAnnealingOptimizer(object):

    def __init__(self, english_text):
        self.scorer = TranslationScorer()
        self.energy = lambda x: self.scorer.score_texts(english_text, x)[0]
        self.temperature = lambda x: math.exp(-x)
        self.k_max = 1000
        self.possible_chars = [ch for ch in string.ascii_letters] + [' ']

    def optimize(self, initial_guess):
        old_energy = self.energy(initial_guess)
        state = initial_guess
        for k in range(self.k_max):
            T = self.temperature(float(k) / self.k_max)
            neighbors = [state + char for char in self.possible_chars] + \
                        [state[:-1] + char for char in self.possible_chars]
            selected_neighbor = random.choice(neighbors)
            curr_energy = self.energy(selected_neighbor)
            delta_energy = curr_energy - old_energy
            if delta_energy > 0:
                state = selected_neighbor
                old_energy = curr_energy
            elif random.uniform(0, 1) <= math.exp(-delta_energy / T):
                state = selected_neighbor
                old_energy = curr_energy

            if k % 50 == 0:
                print('k = ', k, 'state = ', state, ' energy = ', old_energy, ' temperature = ', T)
        return state



if __name__ == '__main__':
    # scorer = TranslationScorer()
    # score = scorer.score_texts('I think that machine translation is very interesting subject.',
    #                      'Ich denke, dass maschinelle Übersetzung ein sehr interessantes Thema ist.')
    # print(score)
    optimizer = SimulatedAnnealingOptimizer('I think that machine translation is very interesting subject.')
    res = optimizer.optimize('Ich')
    print(res)