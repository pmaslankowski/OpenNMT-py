import torch

import numpy as np

from research.beam_optimizer import BeamOptimizer
from research.continuous_optimizer import ContinuousOptimizer
from research.exponentiated_gradient_optimizer import ExponentiatedGradientOptimizer
from research.greedy_optimizer import GreedyOptimizer
from research.scorer import Scorer
from research.utils import load_vocabulary, Tokenizer, OneHotEncoder

if __name__ == '__main__':
    temperature = 1.0
    vocab = load_vocabulary(temperature=temperature)
    tokenizer = Tokenizer()
    scorer = Scorer(temperature=temperature)
    one_hot_encoder = OneHotEncoder(vocab)

    english_sentence = "The victim's brother, Louis Galicia, told ABC station KGO in San Francisco that Frank, previously a line cook in Boston, had landed his dream job as line chef at San Francisco's Sons & Daughters restaurant six months ago."
    german_translation_from_google = 'Ich glaube, dass maschinelle Übersetzung ein sehr interessantes Thema ist.'
    #
    # score = scorer.score_texts(english_sentence, german_translation_from_google)
    # print('Translation scoring')
    # print('English sentence:', english_sentence)
    # print('Translation from google translate to score:', german_translation_from_google)
    # print('Log likehood of translation:', score[0])
    # print()
    #
    # unfinished_translation = 'Ich denke, dass maschinelle Übersetzung ein sehr'
    # english_tokens = tokenizer.tokenize(english_sentence)
    # german_tokens = tokenizer.tokenize(unfinished_translation)
    # next_word_probs = scorer.next_word_probabilities([english_tokens], [german_tokens])
    # val, ind = next_word_probs.topk(3)
    # print('Unfinished translation:', unfinished_translation )
    # print('Top 3 next tokens: ', vocab.itos[ind[0,0].view(-1)], vocab.itos[ind[0,1].view(-1)], vocab.itos[ind[0,2].view(-1)])


    greedy_opt = BeamOptimizer(english_sentence, n_beams=1, temperature=temperature)
    german_tok = greedy_opt.optimize()[:-1]
    print(german_tok)
    print(len(german_tok))
    english_tok = tokenizer.tokenize(english_sentence)
    #german_vecs = one_hot_encoder.encode(torch.tensor([[[vocab.stoi[tok] for tok in german_tok]]]).transpose(0, 2), v=1000.).squeeze().transpose(0, 1)
    print(-scorer.score_tokenized_texts([english_tok], [german_tok], relaxed=False, method='multiplication', normalize=False))
    german_probs = scorer.score_probabilities_for_each_word(english_tok, german_tok)
    #uwaga - przy zmianie zdania trzeba zmienić długość
    german_probs = torch.tensor(german_probs[:-1, :].T)
    
    optimizer = ExponentiatedGradientOptimizer(english_sentence)
    print('Optimization starts...')
    res = optimizer.optimize(init=german_probs, verbose=True)
    print(res)


    # test of score_probabilities_for_each_word function
    # english_tokens = tokenizer.tokenize(english_sentence)
    # german_tokens = tokenizer.tokenize(german_translation_from_google)
    # scores = scorer.score_probabilities_for_each_word(english_tokens, german_tokens)
    # expected_scores = scorer.next_word_probabilities([english_tokens], [german_tokens])[0].detach().numpy()
    # actual_scores = scores[-1, :]
    # if np.sum(expected_scores != actual_scores) == 0:
    #     print('OK')
    # else:
    #     print('FAIL')


    # relaxation:
    # unfinished_translation = 'Ich'
    # english_tokens = tokenizer.tokenize(english_sentence)
    # german_tokens = tokenizer.tokenize(unfinished_translation)
    # german_vec = np.zeros((len(vocab.itos), 1), dtype='float32')
    # german_vec[115,0] = 1.
    # german_vec = torch.tensor(german_vec, requires_grad=True)
    #
    # next_word_probs = scorer.next_word_probabilities([english_tokens], [german_vec], relaxed=True)
    # print(next_word_probs)
    #
    # next_word_probs2 = scorer.next_word_probabilities([english_tokens], [german_tokens], relaxed=False)
    # print(next_word_probs2)

    # relaxation 2:
    # score = scorer.score_tokenized_texts([english_tokens], [german_tokens], relaxed=False)
    # print(score)
    #
    # score2 = scorer.score_tokenized_texts([english_tokens], [german_vec], relaxed=True)
    # print(score2)
    #
    # score2.backward()
    # print(german_vec.grad)
