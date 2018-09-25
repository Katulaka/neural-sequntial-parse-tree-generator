import collections
import numpy as np

import trees
# from beam.search import BeamSearch
# from astar.search import astar_search

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
PAD = "<PAD>"

def flatten(list_of_lists):
    for el in list_of_lists:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

class Parser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            char_vocab,
            label_vocab,
    ):

        self.model = model
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.label_vocab = label_vocab

    def convert_one_sentence(self, tree, is_train):

        def helper(word):
            if word not in (START, STOP):
                    count = self.word_vocab.count(word)
                    if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                        word = UNK
            return word

        batch = [(helper(leaf.word), leaf.tag, tuple(leaf.labels), tuple(leaf.word))
                    for  leaf in tree.leaves()]

        words, tags, labels_in, chars = zip(*batch)

        words = (START,) + words + (STOP,)

        words = tuple([self.word_vocab.index(word) for word in words])

        tags = (START,) + tags + (STOP,)
        tags = tuple([self.tag_vocab.index(tag) for tag in tags])

        chars = (tuple(START),) + chars + (tuple(STOP),)
        chars = [tuple([self.char_vocab.index(c) for c in (tuple(START),) + chars + (tuple(STOP),)])
                    for char in chars]

        labels = [tuple([self.label_vocab.index(l) for l in (START,) + label])
                    for label in labels_in]

        targets = [tuple([self.label_vocab.index(l) for l in label + (STOP,)])
                    for label in labels_in]

        return (words, tags, tuple(labels), tuple(chars), tuple(targets))

    def convert_batch(self, parse_trees, is_train):

        def seq_len(sequences):
            return tuple([len(sequence) for sequence in sequences])

        def pad(sequence, pad, max_len):
            return sequence + (pad,)*(max_len-len(sequence))

        batch = [self.convert_one_sentence(tree, is_train) for tree in parse_trees]

        batch_len = [(len(w), len(t), seq_len(l), seq_len(c)) for w,t,l,c,_ in batch]
        words_len, tags_len, labels_len, chars_len = zip(*batch_len)

        max_len = max(words_len)

        words, tags, labels, chars, targets = zip(*batch)
        words = [pad(word, self.word_vocab.index(PAD), max_len) for word in words]
        tags = [pad(tag, self.tag_vocab.index(PAD), max_len) for tag in tags]

        labels_len = np.reshape([pad((0,)+l, 0, max_len) for l in labels_len], [-1])
        max_labels_len = max(labels_len)
        WORD_PAD = self.label_vocab.index(PAD)
        labels = [pad(((WORD_PAD,),) + label, (WORD_PAD,), max_len) for label in labels]
        labels = [pad(l, WORD_PAD, max_labels_len) for label in labels for l in label ]

        chars_len = np.reshape([pad(c, 0, max_len) for c in chars_len], [-1])
        max_chars_len = max(chars_len)
        CHAR_PAD = self.char_vocab.index(PAD)
        chars = [pad(char, (CHAR_PAD,), max_len) for char in chars]
        chars = [pad(c, CHAR_PAD, max_chars_len) for char in chars for c in char]

        BatchVector = collections.namedtuple('BatchVector', 'input length')
        bv_words = BatchVector(input=np.vstack(words), length=np.array(words_len))
        bv_tags = BatchVector(input=np.vstack(tags), length=np.array(tags_len))
        bv_labels = BatchVector(input=np.vstack(labels), length=labels_len)
        bv_chars = BatchVector(input=np.vstack(chars), length=chars_len)

        bv_targets = np.array(list(flatten(targets)))
        Batch = collections.namedtuple('Batch', 'words tags labels chars targets')
        return Batch(words=bv_words, tags=bv_tags, labels=bv_labels, chars=bv_chars,
                    targets= bv_targets)

    def parse(self, parse_trees, mode):
        if mode == 'train':
            loss, _ = self.model.step(
                            batch=self.convert_batch(parse_trees, is_train=True),
                            output_feed=[self.model.loss, self.model.optimizer],
                            is_train=True)
            return None, loss
        elif mode == 'dev':
            loss = self.model.step(
                        batch=self.convert_batch(parse_trees, is_train=False),
                        output_feed=self.model.loss,
                        is_train=False)
            return None, loss
        else:
            tree = self.predcit(sentence)
            return tree, None
