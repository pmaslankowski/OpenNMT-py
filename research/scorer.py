from onmt.translate.translator import build_translator
from research import consts
from research.utils import Tokenizer


class Scorer(object):

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.translator = build_translator(consts.OPT, report_score=True, create_out_file=False)

    def score_texts(self, english_text, german_translation):
        tokenized_english_text = self.tokenizer.tokenize(english_text)
        tokenized_german_translation = self.tokenizer.tokenize(german_translation)
        return self.translator.score_target(src_data_iter=[' '.join(tokenized_english_text)],
                                       tgt_data_iter=[' '.join(tokenized_german_translation)],
                                       batch_size=consts.OPT.batch_size)

    def score_tokenized_texts(self, english_tok_seq_gen, german_tok_seq_gen):
        return self.translator.score_target(src_data_iter=[' '.join(english_tok_seq) for english_tok_seq in english_tok_seq_gen],
                                            tgt_data_iter=[' '.join(german_tok_seq) for german_tok_seq in german_tok_seq_gen],
                                            batch_size=consts.OPT.batch_size)

    def score(self):
        return self.translator.score_target(src_path=consts.OPT.src,
                             tgt_path=consts.OPT.tgt,
                             src_dir=consts.OPT.src_dir,
                             batch_size=consts.OPT.batch_size)

    def next_word_probabilities(self, english_tok_seq_gen, german_tok_seq_gen):
        return self.translator.next_word_probabilities(
            src_data_iter=[' '.join(english_tok_seq) for english_tok_seq in english_tok_seq_gen],
            tgt_data_iter=[' '.join(german_tok_seq) for german_tok_seq in german_tok_seq_gen],
            batch_size=consts.OPT.batch_size)

    def score_probabilities_for_each_word(self, english_tok_seq, german_tok_seq):
        # [pma]: TODO: add support for batches
        return self.translator.score_probs_for_each_word(
            src_data_iter=[' '.join(english_tok_seq)],
            tgt_data_iter=[' '.join(german_tok_seq)],
            batch_size=consts.OPT.batch_size)