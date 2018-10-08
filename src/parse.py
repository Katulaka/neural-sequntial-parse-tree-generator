import collections
import tensorflow as tf
import numpy as np

import trees
from model.NSPTG_model import NSPTGModel
from beam.search import BeamSearch
from astar.search import astar_search

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
            args,
            tag_vocab,
            word_vocab,
            char_vocab,
            label_vocab,
    ):

        self.config = {
                'ntags' : tag_vocab.size,
                'nwords' : word_vocab.size,
                'nchars': char_vocab.size,
                'nlabels': label_vocab.size,
                'model_path_base': args.model_path_base,
                **{k.split("nn_")[-1] : v for k,v in vars(args).items() if k.startswith("nn_")}
                }

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.label_vocab = label_vocab

    def __call__(self, model_update):

        self.config.update(model_update)
        self.model = NSPTGModel(self.config)
        writer_dir = self.model.model_path_base + '/logs'
        graph = self.model.sess.graph
        self.train_writer = tf.summary.FileWriter(writer_dir + '/train', graph)
        self.dev_writer = tf.summary.FileWriter(writer_dir +  '/dev', graph)

        return self

    def convert_one_sentence(self, sentence, gold, is_train):

        def helper(word):
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            return word

        tags, words, chars = zip(*[(tag, helper(word), tuple(word)) for tag, word in sentence])

        chars = (tuple(START),) + chars + (tuple(STOP),)
        chars = tuple(tuple(self.char_vocab.index(c) for c in ((START),) + ch + ((STOP),))
                        for ch in chars)

        words = (START,) + words + (STOP,)
        words = tuple(self.word_vocab.index(word) for word in words)

        tags = (START,) + tags + (STOP,)
        tags = tuple(self.tag_vocab.index(tag) for tag in tags)

        if gold is not None:
            labels = tuple(tuple(self.label_vocab.index(l) for l in (START,) + label)
                                    for label in gold)

            targets = tuple(tuple(self.label_vocab.index(l) for l in label + (STOP,))
                            for label in gold)

            return (tags, words, chars, labels, targets)

        return (tags, words, chars)


    def convert_batch(self, sentences, gold, is_train):
        is_test = gold is None

        def seq_len(sequences):
            return tuple([len(sequence) for sequence in sequences])

        def pad(sequence, pad, max_len):
            return sequence + (pad,)*(max_len-len(sequence))

        if is_test:
            batch = [self.convert_one_sentence(sentence, None, is_train)
                        for sentence in sentences]
            batch_len = [(len(t), len(w), seq_len(c)) for t,w,c,in batch]
            tags_len, words_len, chars_len = zip(*batch_len)
            tags, words, chars = zip(*batch)
        else:
            batch = [self.convert_one_sentence(sentence, g, is_train)
                            for sentence, g in zip(sentences, gold)]
            batch_len = [(len(t), len(w), seq_len(c), seq_len(l)) for t,w,c,l,_ in batch]
            tags_len, words_len, chars_len, labels_len = zip(*batch_len)
            tags, words, chars, labels, targets = zip(*batch)

        max_len = max(words_len)

        words = [pad(word, self.word_vocab.index(PAD), max_len) for word in words]
        tags = [pad(tag, self.tag_vocab.index(PAD), max_len) for tag in tags]

        chars_len = np.reshape([pad(c, 0, max_len) for c in chars_len], [-1])
        max_chars_len = max(chars_len)
        CHAR_PAD = self.char_vocab.index(PAD)
        chars = [pad(char, (CHAR_PAD,), max_len) for char in chars]
        chars = [pad(c, CHAR_PAD, max_chars_len) for char in chars for c in char]

        if not is_test:
            labels_len = [pad((0,) + l, 0, max_len) for l in labels_len]
            labels_len = np.reshape(labels_len, [-1])
            max_labels_len = max(labels_len)
            LABEL_PAD = self.label_vocab.index(PAD)
            labels = [pad(((LABEL_PAD,),) + label, (LABEL_PAD,), max_len) for label in labels]
            labels = [pad(l, LABEL_PAD, max_labels_len) for label in labels for l in label ]

        BatchVector = collections.namedtuple('BatchVector', 'input length')
        bv_tags = BatchVector(input=np.vstack(tags), length=np.array(tags_len))
        bv_words = BatchVector(input=np.vstack(words), length=np.array(words_len))
        bv_chars = BatchVector(input=np.vstack(chars), length=chars_len)
        bv_labels = None if is_test else BatchVector(input=np.vstack(labels), length=labels_len)
        bv_targets = None if is_test else np.array(list(flatten(targets)))

        Batch = collections.namedtuple('Batch', 'tags words chars labels targets')
        return Batch(
                tags=bv_tags,
                words=bv_words,
                chars=bv_chars,
                labels=bv_labels,
                targets= bv_targets
                )

    def parse(self, sentences, gold, mode, predict_parms=None):
        if mode == 'train':
            output_feed = [self.model.loss, self.model.optimizer]
            batch = self.convert_batch(sentences, gold, is_train=True)
            loss,  _ = self.model.step(
                                batch=batch,
                                output_feed=output_feed,
                                is_train=True
                            )

            return loss

        elif mode == 'dev':
            loss = self.model.step(
                        batch=self.convert_batch(sentences, gold, is_train=False),
                        output_feed=self.model.loss,
                        is_train=False)
            return loss
        else:
            start = self.label_vocab.index(START)
            stop = self.label_vocab.index(STOP)
            astar_parms = predict_parms['astar_parms']
            enc_bv = self.convert_batch(sentences, gold, is_train=False)
            enc_state = self.model.encode_top_state(enc_bv)[1:enc_bv.words.length - 1]
            for beam_size in predict_parms['beam_parms']:
                hyps = BeamSearch(start, stop, beam_size).beam_search(
                                                    enc_state,
                                                    self.model.decode_topk,
                                                    )

                grid = []
                for i, (leaf_hyps, leaf) in enumerate(zip(hyps, sentences)):
                    row = []
                    for hyp in leaf_hyps:
                        labels = np.array(self.label_vocab.values)[hyp[0]].tolist()
                        partial_tree = trees.LeafMyParseNode(i, *leaf).deserialize(labels)
                        if partial_tree is not None:
                            row.append((partial_tree, hyp[1]))
                    grid.append(row)


                nodes = astar_search(grid, self.keep_valence_value, astar_parms)

                if nodes != []:
                    return nodes[0].trees[0], None

            children = [trees.LeafMyParseNode(i, *leaf) for i,leaf in enumerate(sentence)]
            tree = trees.InternalMyParseNode('S', children)

            return tree, None

    def log(self, value, is_train):

        writer = self.train_writer if is_train else self.dev_writer
        loss = tf.Summary()
        global_step = self.model.sess.run(self.model.global_step)
        loss.value.add(tag="loss", simple_value=value)
        writer.add_summary(loss, global_step)
        writer.flush()
