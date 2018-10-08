"""Beam search module.

Beam search takes the top K results from the model, predicts the K results for
each of the previous K result, getting K*K results. Pick the top K results from
K*K results, and start over again until certain number of results are fully
decoded.
"""

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf

class Hypothesis(object):
    """Defines a hypothesis during beam search."""

    def __init__(self, tokens, prob, state, score=None):
        """Hypothesis constructor.

        Args:
          tokens: start tokens for decoding.
          prob: prob of the start tokens, usually 1.
          state: decoder initial states.
          score: decoder intial score.
        """
        self.tokens = tokens
        self.prob = prob
        self.state = state
        self.score = math.log(prob[-1]) if score is None else score

    def extend_(self, token, prob, new_state):
        """Extend the hypothesis with result from latest step.

        Args:
          token: latest token from decoding.
          prob: prob of the latest decoded tokens.
          new_state: decoder output state. Fed to the decoder for next step.
        Returns:
          New Hypothesis with the results from latest step.
        """
        tokens = self.tokens + [token]
        probs = self.prob + [prob]
        score = self.score + math.log(prob)
        return Hypothesis(tokens, probs, new_state, score)

    @property
    def latest_token(self):
        return self.tokens[-1]

    def __str__(self):
        return ('Hypothesis(prob = {:4f}, tokens = {})'.format(
                        self.prob, self.tokens)
                )


class BeamSearch(object):
    """Beam search."""

    def __init__(self, start_token, end_token, beam_size, max_steps=28):
        """Creates BeamSearch object.

        Args:
          beam_size: int.
          start_token: int, id of the token to start decoding with
          end_token: int, id of the token that completes an hypothesis
          max_steps: int, upper limit on the size of the hypothesis
        """
        self._beam_size = beam_size
        self._start_token = start_token
        self._end_token = end_token
        self._max_steps = max_steps

    def beam_search(self, enc_state, decode_topk, tags):
        """Performs beam search for decoding.

         Args:

            enc_state: ndarray of shape (enc_length, 1),
                        the document ids to encode
            decode_topk: ndarray of shape (1), the length of the sequnce


         Returns:
            hyps: list of Hypothesis, the best hypotheses found by beam search,
                    ordered by score
         """

        hyps_per_sentence = []
        #iterate over words in seq
        for dec_in, tag in zip(enc_state, tags):
            c_cell = np.expand_dims(dec_in, axis=0)
            h_cell = np.expand_dims(np.zeros_like(dec_in), axis=0)
            dec_in_state = tf.contrib.rnn.LSTMStateTuple(c_cell, h_cell)
            complete_hyps = []
            hyps = [Hypothesis([self._start_token], [1.0], dec_in_state)]
            _, _, new_state = decode_topk(
                                latest_tokens = [[self._start_token]],
                                init_states = dec_in_state,
                                enc_state = [enc_state])
            hyps = [hyps[0].extend_(tag, 1.0, new_state)]
            for steps in xrange(self._max_steps):
                if hyps != []:
                    # Extend each hypothesis.
                    # The first step takes the best K results from first hyps.
                    # Following steps take the best K results from K*K hyps.
                    all_hyps = []
                    latest_token = [[hyp.latest_token] for hyp in hyps]
                    c_cell = np.array([np.squeeze(hyp.state[0]) for hyp in hyps])
                    h_cell = np.array([np.squeeze(hyp.state[1]) for hyp in hyps])

                    states = tf.contrib.rnn.LSTMStateTuple(c_cell, h_cell)
                    ids, probs, new_state = decode_topk(
                                                latest_token,
                                                states,
                                                [enc_state],
                                                self._beam_size)

                    for k, hyp in enumerate(hyps):
                        c_cell = np.expand_dims(new_state[0][k], axis=0)
                        h_cell = np.expand_dims(new_state[1][k], axis=0)
                        state = tf.contrib.rnn.LSTMStateTuple(c_cell, h_cell)
                        for j in xrange(self._beam_size):
                            all_hyps.append(
                                    hyp.extend_(ids[k][j], probs[k][j], state)
                                    )
                    hyps = []

                    for h in self.best_hyps(all_hyps):
                        # Filter and collect any hypotheses that have the end token.
                        if h.latest_token == self._end_token and len(h.tokens)>2:
                            # Pull the hypothesis off the beam
                            #if the end token is reached.
                            complete_hyps.append(h)
                        elif h.latest_token == self._end_token:
                            pass
                        elif len(complete_hyps) >= self._beam_size \
                            and h.score < min(complete_hyps, key=lambda h: h.score).score:
                            pass
                        else:
                            # Otherwise continue to the extend the hypothesis.
                            hyps.append(h)
            hyps_per_word = self.best_hyps(complete_hyps)
            hyps_per_sentence.append([(h.tokens[2:-1], h.score) for h in hyps_per_word])

        return hyps_per_sentence

    def best_hyps(self, hyps):
        """return top <beam_size> hyps.

        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A sub list of top <beam_size> hyps.
        """
        return sorted(hyps, key=lambda h: h.score, reverse=True)[:self._beam_size]
