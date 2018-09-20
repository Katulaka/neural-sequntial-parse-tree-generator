from subprocess import Popen, DEVNULL, PIPE
import argparse

import evaluate
import trees

def get_dependancies(fin, penn_path):
    """ Creates dependancy dictionary for each intput file"""

    command = 'java -jar {} < {} -splitSlash=false'.format(penn_path, fin)
    proc = Popen(command, shell=True, stdout=PIPE, stderr=DEVNULL)
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

def run_train(args):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = trees.load_trees(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Processing trees for training...")
    dependancies = get_dependancies(args.train_path, args.penn_path)
    train_parse = [tree.convert(dep)(args.keep_valence_value)
                        for tree, dep in zip(train_treebank, dependancies)]

    def loopback(parse, astar_parms = [1, 100., 10., 0.2]):
        beams = [[(trees.LeafMyParseNode(i, l.tag, l.word).deserialize(l.labels),  1.)]
                        for i, l in enumerate(parse.leaves())]
        return astar_search(beams, False, astar_parms, 0)

    import evaluate
    start = 0
    end = 39832
    predicted = []
    for parse in train_parse[start:end]:
        predicted.append(loopback(parse).convert())
    print (evaluate.evalb("../../POST/EVALB/", train_treebank[start:end], predicted))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--train-path", default="data/02-21.10way.clean")
    subparser.add_argument("--dev-path", default="data/22.auto.clean")
    subparser.add_argument("--penn-path", default="utils/pennconverter.jar")
    subparser.add_argument("--keep-valence-value", action="store_true")


    args = parser.parse_args()
    args.callback(args)

if __name__ == "__main__":
    main()
