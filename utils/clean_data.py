"""Clean the ptb dataset for our model"""

from nltk.corpus import BracketParseCorpusReader as reader
import argparse
import os

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

def create_clean_data(src_dir):
    """ Converst raw PTB data into clean train/dev/test data"""
    dataset = dict()
    for directory, dirnames, filenames in os.walk(src_dir):
        if directory[-1].isdigit() and directory[-2:] not in ['00','01','24']:
            data_key = directory.split('/')[-1]
            for fname in sorted(filenames):
                path_f = '/'.join([directory, fname]).split('/')
                # path_f_s = path_f.split('/')
                r = reader('/'.join(path_f[:-2]), '/'.join(path_f[-2:]))
                trees = simplify(remove_traces(list(r.parsed_sents())))
                for t in trees:
                    t_lin = ' '.join(t.pformat().replace('\n', '').split())
                    dataset.setdefault(data_key, []).append(t_lin)

    for r in [(2,22), (22,23), (23,24)]:
        fname = 'data/{:02d}-{}.clean'.format(r[0],r[1]-1)
        with open(fname, 'w') as f:
            for k in range(*r):
                for l in dataset[str(k).zfill(2)]:
                    f.write("%s\n" % l)

if __name__ == "__main__":
    create_clean_data('../raw_data/wsj/')
