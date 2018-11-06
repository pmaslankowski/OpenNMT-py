import argparse
import onmt
import onmt.opts as opts
import sentencepiece as spm
from research import consts


def load_vocabulary():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, _, _ = \
        onmt.model_builder.load_test_model(consts.OPT, dummy_opt.__dict__)

    return fields['tgt'].vocab


class Tokenizer(object):

    def __init__(self):
        self.sentencepiece_tokenizer = spm.SentencePieceProcessor()
        self.sentencepiece_tokenizer.Load(consts.SENTENCEPIECE_MODEL_PATH)

    def tokenize(self, text):
        return self.sentencepiece_tokenizer.EncodeAsPieces(text)