import argparse
import itertools
import numpy as np
import os
import pickle
from subprocess import Popen, DEVNULL, PIPE
import time

import evaluate
import parse
import trees
import vocabulary


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def get_dependancies(fin, penn_path):
    """ Creates dependancy dictionary for each intput file"""

    command = 'java -jar {} < {} -splitSlash=false'.format(penn_path, fin)
    # proc = Popen(command, shell=True, stdout=PIPE, stderr=DEVNULL)
    proc = Popen(command, shell=True, stdout=PIPE)
    results = proc.stdout.readlines()
    dependancies = []
    dependancy = []
    for res in results:
        res = res.decode('utf8')
        if res == '\n':
            dependancies.append(dependancy)
            dependancy = []
        else:
            dependancy.append(int(res.split()[6]))
    return dependancies

def loopback(parse, astar_parms, keep_valence_value):
    beams = [[(trees.LeafMyParseNode(i, l.tag, l.word).deserialize(l.labels),  1.)]
                    for i, l in enumerate(parse.leaves())]
    return astar_search(beams, keep_valence_value, astar_parms, 0)

def run_train(args):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = trees.load_trees(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading developing trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} developing examples.".format(len(dev_treebank)))


    print("Processing dependancies for training...")
    dependancies = get_dependancies(args.train_path, args.penn_path)
    print("Processing trees for training...")
    train_parse = [tree.convert(dep)(args.keep_valence_value)
                        for tree, dep in zip(train_treebank, dependancies)]

    print("Processing dependancies for developing...")
    dependancies = get_dependancies(args.dev_path, args.penn_path)
    print("Processing trees for developing...")
    dev_parse = [tree.convert(dep)(args.keep_valence_value)
                        for tree, dep in zip(dev_treebank, dependancies)]

    print("Initializing model...")

    parser_path = os.path.join(args.model_path_base, 'parser.pkl')
    if not os.path.exists(parser_path):

        print("Constructing vocabularies...")

        tag_vocab = vocabulary.Vocabulary()
        tag_vocab.index(parse.PAD)
        tag_vocab.index(parse.START)
        tag_vocab.index(parse.STOP)

        word_vocab = vocabulary.Vocabulary()
        word_vocab.index(parse.PAD)
        word_vocab.index(parse.START)
        word_vocab.index(parse.STOP)
        word_vocab.index(parse.UNK)

        char_vocab = vocabulary.Vocabulary()
        char_vocab.index(parse.PAD)
        char_vocab.index(parse.START)
        char_vocab.index(parse.STOP)
        for c in parse.START+parse.STOP+parse.UNK:
            char_vocab.index(c)

        label_vocab = vocabulary.Vocabulary()
        label_vocab.index(parse.PAD)
        label_vocab.index(parse.START)
        label_vocab.index(parse.STOP)

        for tree in train_parse:
            nodes = [tree]
            while nodes:
                node = nodes.pop()
                if isinstance(node, trees.InternalMyParseNode):
                    nodes.extend(reversed(node.children))
                else:
                    for l in node.labels:
                        label_vocab.index(l)
                    for c in node.word:
                        char_vocab.index(c)
                    tag_vocab.index(node.tag)
                    word_vocab.index(node.word)

        tag_vocab.freeze()
        word_vocab.freeze()
        char_vocab.freeze()
        label_vocab.freeze()

        # os.makedirs(args.model_path_base)
        parser = parse.Parser(
                    args,
                    tag_vocab,
                    word_vocab,
                    char_vocab,
                    label_vocab
                )
        with open(parser_path, 'wb') as f:
            pickle.dump(parser, f, pickle.HIGHEST_PROTOCOL)

    else:

        with open(parser_path, 'rb') as f:
            parser = pickle.load(f)

    parser = parser({'mode': args.mode, 'gpu_id': args.gpu_id})

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_model_path = None
    best_dev_loss = np.inf

    start_time = time.time()

    def check_dev():
        nonlocal best_dev_loss
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        total_losses = []
        for start_index in range(0, len(dev_parse), args.batch_size):
            batch_losses = []
            parse_trees = dev_parse[start_index:start_index + args.batch_size]

            sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()]
                                for tree in parse_trees]

            gold = [[tuple(leaf.labels) for leaf in tree.leaves()]
                                for tree in parse_trees]

            batch_loss = parser.parse(sentences, gold, mode='dev')
            total_losses.append(batch_loss)

            print(
                "batch {:,}/{:,} "
                "batch-loss {:.4f} "
                "dev-elapsed {} "
                "total-elapsed {}".format(
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(dev_parse) / args.batch_size)),
                    total_losses[-1],
                    format_elapsed(dev_start_time),
                    format_elapsed(start_time),
                )
            )

        dev_loss = np.mean(total_losses)

        parser.log(value=dev_loss, is_train=False)

        print(
            "dev-loss {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_loss,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_loss < best_dev_loss:

            best_dev_loss = dev_loss
            best_dev_model_path = "dev={:.4f}".format(dev_loss)
            print("Saving new best model to {}/checkpoints/{}...".format(
                                                        args.model_path_base,
                                                        best_dev_model_path))
            parser.model.save(best_dev_model_path)

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        # np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            parse_trees = train_parse[start_index:start_index + args.batch_size]

            sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()]
                                for tree in parse_trees]

            gold = [[tuple(leaf.labels) for leaf in tree.leaves()]
                                for tree in parse_trees]

            batch_loss_value = parser.parse(sentences, gold, mode='train')

            parser.log(value=batch_loss_value, is_train=True)

            total_processed += len(parse_trees)
            current_processed += len(parse_trees)

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()

def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))

    parser_path = os.path.join(args.model_path_base, 'parser.pkl')
    if os.path.exists(parser_path):
        with open(parser_path, 'rb') as f:
            parser = pickle.load(f)
        parser = parser({'mode': args.mode, 'gpu_id': args.gpu_id})
    else:
        print("Couldn't load {}".format(parser_path))

    print("Parsing test sentences...")

    start_time = time.time()

    test_predicted = []
    predict_parms = {'astar_parms': args.astar_parms, 'beam_parms':args.beam_size}
    for i, tree in  enumerate(test_treebank):
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
        prediction_start_time = time.time()
        predicted, _ = parser.parse(sentence, None, mode='test', predict_parms=predict_parms)
        print(
            "processed {:,}/{:,} "
            "prediction-elapsed {} "
            "total-elapsed {}".format(
                i+1,
                len(test_treebank),
                format_elapsed(prediction_start_time),
                format_elapsed(start_time),
            )
        )
        test_predicted.append(predicted.convert())

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train, mode='train')
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--train-path", default="data/02-21.clean")
    subparser.add_argument("--dev-path", default="data/22-22.clean")
    subparser.add_argument("--penn-path", default="utils/pennconverter.jar")
    subparser.add_argument("--keep-valence-value", action="store_true")
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--batch-size", type=int, default=32)
    subparser.add_argument("--tag-dim", type=int, default=32, dest="nn_tag_dim")
    subparser.add_argument("--word-dim", type=int, default=64, dest="nn_word_dim")
    subparser.add_argument("--char-dim", type=int, default=32, dest="nn_char_dim")
    subparser.add_argument("--label-dim", type=int, default=32, dest="nn_label_dim")
    subparser.add_argument("--h-word", type=int, default=128, dest="nn_h_word")
    subparser.add_argument("--h-char", type=int, default=32, dest="nn_h_char")
    subparser.add_argument("--h-label", type=int, default=384, dest="nn_h_label")
    subparser.add_argument("--attention-dim", type=int, default=256, dest="nn_attention_dim")
    subparser.add_argument("--projection-dim", type=int, default=128, dest="nn_projection_dim")
    subparser.add_argument('--dropouts', nargs='+', default=[0.2, 0.5], type=float, dest="nn_dropouts")
    subparser.add_argument('--n-layers', type=int, default=2, dest="nn_n_layers")
    subparser.add_argument('--layer-norm', action='store_true', dest="nn_layer_norm")
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--gpu-id", type=int, default=0)

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test, mode='test')
    subparser.add_argument("--test-path", default="data/23-23.clean")
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--gpu-id", type=int, default=0)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--astar-parms", type=float, nargs=4, default=[1, 60., 1, 0.2])
    subparser.add_argument("--beam-size", type=int, nargs='+', default=[5])

    args = parser.parse_args()
    args.callback(args)

if __name__ == "__main__":
    main()
