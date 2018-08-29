"""Read, split and save the ptb dataset for our model"""

from tree_t import TreeT
from nltk.corpus import BracketParseCorpusReader as reader
import collections
import argparse
import os



parser = argparse.ArgumentParser()
parser.add_argument('--ds_file', default='dataset.pkl', help="File containing the dataset")
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")

#Parameters for dataset
TRAIN_DIRS = (2,22)
DEV_DIRS = (22,23)
TEST_DIRS = (23,24)


def remove_traces(ts): # Remove traces and null elements
    for t in ts:
        for ind, leaf in reversed(list(enumerate(t.leaves()))):
            postn = t.leaf_treeposition(ind)
            parentpos = postn[:-1]
            if leaf.startswith("*") or t[parentpos].label() == '-NONE-':
                while parentpos and len(t[parentpos]) == 1:
                    postn = parentpos
                    parentpos = postn[:-1]
                print(t[postn], "will be deleted")
                del t[postn]
    return ts

def simplify(ts): # Simplify tags
    for t in ts:
        for s in t.subtrees():
            tag = s.label()
            if tag not in ['-LRB-', '-RRB-', '-LCB-', '-RCB-', '-NONE-']:
                if '-' in tag or '=' in tag or '|' in tag:
                    simple = tag.split('-')[0].split('=')[0].split('|')[0]
                    s.set_label(simple)
                    print('substituting', simple, 'for', tag)
    return ts

def get_dependancies(fin, path_penn):
    """ Creates dependancy dictionary for each intput file"""

    cmd = 'java -jar {} < {} -splitSlash=false'.format(path_penn, fin)
    depends = os.popen(cmd).read().split('\n')
    assert depends[-1] == ''
    depends_l = []
    while depends:
        c_idx = depends.index('')
        c_dep = depends[:c_idx]
        dep_dict = dict([(int(l.split()[0]), int(l.split()[6])) for l in c_dep])
        depends_l.append(dep_dict)
        depends = depends[c_idx+1:]


    return depends_l

def create_dataset(src_dir, path_penn):
    """ Converst raw PTB data into dictinary of:
    - gold : the origninal PTB data after clean up
    - words : the sentence
    - pos : parts of speech
    - chars : each word in the sentence is split into characters
    - tags : the modified tags capturing the tree structure seq
    """
    dataset = dict()
    fields = ['gold', 'words', 'pos', 'chars', 'tags']
    Entry = collections.namedtuple('entry', fields)
    for directory, dirnames, filenames in os.walk(src_dir):
        if directory[-1].isdigit() and directory[-2:] not in ['00','01','24']:
            for fname in sorted(filenames):
                path_f = '/'.join([directory, fname])
                deps = get_dependancies(path_f, path_penn)
                path_f_s = path_f.split('/')
                r = reader('/'.join(path_f_s[:-2]), '/'.join(path_f_s[-2:]))
                trees = simplify(remove_traces(list(r.parsed_sents())))
                gold, words, pos, chars, tags = [], [], [], [], []
                for i, t in enumerate(trees):
                    gold.append(' '.join(t.pformat().replace('\n', '').split()))
                    tup_w, tup_pos = zip(*t.pos())
                    words.append(list(tup_w))
                    pos.append(list(tup_pos))
                    chars.append([list(w) for w in words[-1]])
                    gold_ = gold[-1].replace('(', ' ( ').replace(')', ' ) ').split()
                    max_id = len(pos[-1]) + 1
                    tags.append(TreeT().from_ptb_to_tag(gold_, max_id, deps[i]))
                data_dict = Entry(gold, words, pos, chars, tags)._asdict()
                data_key = fname.split('_')[-1].split('.')[0]
                dataset.setdefault(data_key, {}).update(data_dict)
    return dataset

def split_dataset(dataset):
    """ Split dataset into 3 parts train, dev and test"""

    fields = ['train', 'dev', 'test']
    Entry = collections.namedtuple('entry', fields)
    split_dict = Entry(TRAIN_DIRS, DEV_DIRS, TEST_DIRS)._asdict()

    ds = {}
    #iterate over the split
    for k, v in split_dict.items():
        ds_s = {}
        # iterate over the results from the files in dataset
        for d_k, d_v in sorted(dataset.items()):
            if int(d_k[:2]) in range(*v):
                # iterate over the different types of inputs e.g. chars
                for e_k, e_v in d_v.items():
                    ds_s.setdefault(e_k, []).extend(e_v)
        ds_s['size'] = len(ds_s[e_k])
        ds.setdefault(k, {}).update(ds_s)
    return ds

if __name__ == "__main__":

    args = parser.parse_args()
    path_data = '../raw_data/wsj/'
    path_penn = 'utils/pennconverter.jar'
    path_s_dataset = os.path.join(args.data_dir, 's_'+args.ds_file)
    path_dataset = os.path.join(args.data_dir, args.ds_file)

    if os.path.exists(path_s_dataset):
        with open(path_s_dataset, 'rb') as f:
            s_dataset = pickle.load(f)
    else:
        if os.path.exists(path_dataset):
            with open(path_dataset, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = create_dataset(path_data, path_penn)
            with open(path_dataset, 'wb') as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        s_dataset = split_dataset(dataset)
        with open(path_s_dataset, 'wb') as f:
            pickle.dump(s_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    return s_dataset
