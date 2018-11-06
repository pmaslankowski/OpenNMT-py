import argparse
import math
import random
import string

import numpy as np
import sentencepiece as spm

import onmt
import onmt.opts as opts
from onmt.translate.translator import build_translator


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
            'verbose': True, 'log_file': '', 'attn_debug': False, 'dump_beam': '', 'n_best': 1, 'batch_size': 100, 'gpu': -1,
            'sample_rate': 16000, 'window_size': 0.02, 'window_stride': 0.01, 'window': 'hamming', 'image_channel_size': 3
        })


def load_vocabulary():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, _, _ = \
        onmt.model_builder.load_test_model(OPT, dummy_opt.__dict__)

    return fields['tgt'].vocab

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

    def score_tokenized_texts(self, english_tok_seq_gen, german_tok_seq_gen):
        return self.translator.score_target(src_data_iter=[' '.join(english_tok_seq) for english_tok_seq in english_tok_seq_gen],
                                            tgt_data_iter=[' '.join(german_tok_seq) for german_tok_seq in german_tok_seq_gen],
                                            batch_size=self.opt.batch_size)

    def score(self):
        return self.translator.score_target(src_path=self.opt.src,
                             tgt_path=self.opt.tgt,
                             src_dir=self.opt.src_dir,
                             batch_size=self.opt.batch_size)

    def next_word_probabilities(self, english_tok_seq_gen, german_tok_seq_gen):
        return self.translator.next_word_probabilities(
            src_data_iter=[' '.join(english_tok_seq) for english_tok_seq in english_tok_seq_gen],
            tgt_data_iter=[' '.join(german_tok_seq) for german_tok_seq in german_tok_seq_gen],
            batch_size=self.opt.batch_size)


class Tokenizer(object):

    def __init__(self):
        self.sentencepiece_tokenizer = spm.SentencePieceProcessor()
        self.sentencepiece_tokenizer.Load(SENTENCEPIECE_MODEL_PATH)

    def tokenize(self, text):
        return self.sentencepiece_tokenizer.EncodeAsPieces(text)


class SimulatedAnnealingOptimizer(object):

    def __init__(self, english_text):
        self.tokenize_english_text(english_text)
        self.opt = OPT
        self.specials = ['<blank>', '<unk>', 0, '<s>', '</s>']
        self.scorer = TranslationScorer()
        self.energy = lambda x: self.scorer.score_tokenized_texts([self.english_tok_seq for _ in range(len(x))], x)
        self.temperature = lambda x: math.exp(-x)
        self.k_max = 10000
        self.possible_chars = [ch for ch in string.ascii_letters] + [' ']
        self.vocab = load_vocabulary()

    def tokenize_english_text(self, english_text):
        sentencepiece_processor = spm.SentencePieceProcessor()
        sentencepiece_processor.Load(SENTENCEPIECE_MODEL_PATH)
        self.english_tok_seq = sentencepiece_processor.EncodeAsPieces(english_text)

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



if __name__ == '__main__':
    scorer = TranslationScorer()
    # score = scorer.score_texts('I think that machine translation is very interesting subject.',
    #                      'Ich denke, dass maschinelle Übersetzung ein sehr interessantes Thema ist.')
    # print(score)

    vocab = load_vocabulary()
    tokenizer = Tokenizer()
    english_tokens = tokenizer.tokenize('I think that machine translation is very interesting subject.')
    german_tokens = tokenizer.tokenize('Ich denke, dass maschinelle Übersetzung ein sehr')
    next_word_probs = scorer.next_word_probabilities([english_tokens], [german_tokens])
    val, ind = next_word_probs.topk(1)
    print(vocab.itos[ind.view(-1)])

    # optimizer = SimulatedAnnealingOptimizer('I think that machine translation is very interesting subject.')
    # res = optimizer.optimize()
    # print(res)