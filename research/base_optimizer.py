from research.scorer import Scorer
from research.utils import load_vocabulary, Tokenizer


class BaseOptimizer(object):

    def __init__(self, english_text):
        self.english_tok_seq = self.tokenize_english_text(english_text)
        self.scorer = Scorer()
        self.vocab = load_vocabulary()
        self.vocab_size = len(self.vocab)

    def tokenize_english_text(self, english_text):
        tokenizer = Tokenizer()
        return tokenizer.tokenize(english_text)