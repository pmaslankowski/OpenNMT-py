from research.scorer import Scorer
from research.utils import load_vocabulary, Tokenizer

if __name__ == '__main__':
    scorer = Scorer()
    score = scorer.score_texts('I think that machine translation is very interesting subject.',
                         'Ich denke, dass maschinelle Übersetzung ein sehr interessantes Thema ist.')
    print(score)


    vocab = load_vocabulary()
    tokenizer = Tokenizer()
    english_tokens = tokenizer.tokenize('I think that machine translation is very interesting subject.')
    german_tokens = tokenizer.tokenize('Ich denke, dass maschinelle Übersetzung ein sehr')
    next_word_probs = scorer.next_word_probabilities([english_tokens], [german_tokens])
    val, ind = next_word_probs.topk(1)
    print(vocab.itos[ind.view(-1)])
