import argparse

import sentencepiece as spm

import onmt
from onmt.translate.translator import build_translator
import onmt.opts as opts
import random
import string
import math


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


MODEL_PATH = '../data/transformer/averaged-10-epoch.pt'
SENTENCEPIECE_MODEL_PATH = '../data/transformer/sentencepiece.model'
SOURCE_PATH = 'example.atok'
TARGET_PATH = 'example_target.atok'
OUTPUT_PATH = 'example_output.atok'
OPT = AttrDict({
            'models': [MODEL_PATH], 'data_type': 'text', 'src': SOURCE_PATH, 'src_dir': '',
            'tgt': TARGET_PATH, 'output': OUTPUT_PATH, 'report_bleu': False, 'report_rouge': False,
            'dynamic_dict': False, 'share_vocab': False, 'fast': False, 'beam_size': 5, 'min_length': 0, 'max_length': 100,
            'max_sent_length': None, 'stepwise_penalty': False, 'length_penalty': 'none', 'coverage_penalty': 'none',
            'alpha': 0.0, 'beta': -0.0, 'block_ngram_repeat': 0, 'ignore_when_blocking': [], 'replace_unk': False,
            'verbose': True, 'log_file': '', 'attn_debug': False, 'dump_beam': '', 'n_best': 1, 'batch_size': 30, 'gpu': -1,
            'sample_rate': 16000, 'window_size': 0.02, 'window_stride': 0.01, 'window': 'hamming', 'image_channel_size': 3
        })


class TranslationScorer(object):

    def __init__(self):
        self.opt = OPT
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

    def score_tokenized_texts(self, english_tok_seq, german_tok_seq):
        return self.translator.score_target(src_data_iter=[' '.join(english_tok_seq)],
                                            tgt_data_iter=[' '.join(german_tok_seq)],
                                            batch_size=self.opt.batch_size)

    def score(self):
        return self.translator.score_target(src_path=self.opt.src,
                             tgt_path=self.opt.tgt,
                             src_dir=self.opt.src_dir,
                             batch_size=self.opt.batch_size)



class SimulatedAnnealingOptimizer(object):

    def __init__(self, english_text):
        self.tokenize_english_text(english_text)
        self.opt = OPT
        self.scorer = TranslationScorer()
        self.energy = lambda x: self.scorer.score_tokenized_texts(self.english_tok_seq, x)[0]
        self.temperature = lambda x: math.exp(-x)
        self.k_max = 10000
        self.possible_chars = [ch for ch in string.ascii_letters] + [' ']
        self.load_vocabulary()

    def tokenize_english_text(self, english_text):
        sentencepiece_processor = spm.SentencePieceProcessor()
        sentencepiece_processor.Load(SENTENCEPIECE_MODEL_PATH)
        self.english_tok_seq = sentencepiece_processor.EncodeAsPieces(english_text)

    def load_vocabulary(self):
        dummy_parser = argparse.ArgumentParser(description='train.py')
        opts.model_opts(dummy_parser)
        dummy_opt = dummy_parser.parse_known_args([])[0]

        fields, _, _= \
            onmt.model_builder.load_test_model(self.opt, dummy_opt.__dict__)

        self.vocab = fields['tgt'].vocab


    def optimize(self):
        state = [random.choice(self.vocab.itos)]
        print('initial state = ', state)
        old_energy = self.energy(state)
        for k in range(self.k_max):
            T = self.temperature(float(k) / self.k_max)
            next_word = random.choice(self.vocab.itos)
            neighbor_state = state + [next_word]
            if random.uniform(0, 1) <= 0.5:
                neighbor_state = state[:-1] + [next_word]

            curr_energy = self.energy(neighbor_state)
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



if __name__ == '__main__':
    # scorer = TranslationScorer()
    # score = scorer.score_texts('I think that machine translation is very interesting subject.',
    #                      'Ich denke, dass maschinelle Übersetzung ein sehr interessantes Thema ist.')
    # print(score)
    optimizer = SimulatedAnnealingOptimizer('I think that machine translation is very interesting subject.')
    res = optimizer.optimize()
    print(res)