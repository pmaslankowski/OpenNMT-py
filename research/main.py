import numpy as np

from research.beam_optimizer import BeamOptimizer
from research.scorer import Scorer
from research.utils import load_vocabulary, Tokenizer

if __name__ == '__main__':
    vocab = load_vocabulary()
    tokenizer = Tokenizer()
    scorer = Scorer()

    english_sentence = 'I think that machine translation is very interesting subject.'
    german_translation_from_google = 'Ich denke, dass maschinelle Übersetzung ein sehr interessantes Thema ist.'
    #
    # score = scorer.score_texts(english_sentence, german_translation_from_google)
    # print('Translation scoring')
    # print('English sentence:', english_sentence)
    # print('Translation from google translate to score:', german_translation_from_google)
    # print('Log likehood of translation:', score[0])
    # print()
    #
    # unfinished_translation = 'Ich denke, dass maschinelle Übersetzung ein sehr'
    #english_tokens = tokenizer.tokenize(english_sentence)
    #german_tokens = tokenizer.tokenize(unfinished_translation)
    # next_word_probs = scorer.next_word_probabilities([english_tokens], [german_tokens])
    # val, ind = next_word_probs.topk(3)
    # print('Unfinished translation:', unfinished_translation )
    # print('Top 3 next tokens: ', vocab.itos[ind[0,0].view(-1)], vocab.itos[ind[0,1].view(-1)], vocab.itos[ind[0,2].view(-1)])

    # optimizer = BeamOptimizer(english_sentence)
    # print('Optimization starts...')
    # res = optimizer.optimize()
    # print(res)

    # test of score_probabilities_for_each_word function
    english_tokens = tokenizer.tokenize(english_sentence)
    german_tokens = tokenizer.tokenize(german_translation_from_google)
    scores = scorer.score_probabilities_for_each_word(english_tokens, german_tokens)
    expected_scores = scorer.next_word_probabilities([english_tokens], [german_tokens])[0].detach().numpy()
    actual_scores = scores[-1, :]
    if np.sum(expected_scores != actual_scores) == 0:
        print('OK')
    else:
        print('FAIL')