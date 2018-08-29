""" Main tree class used in this project """

import copy
from treelib import Tree


#Parameters for of the augmented tags
R = '}'
L = '{'
CR = '>'
CL = '<'
UP = '+'
NA = '|'
ANY = '*'


#Attributes of TreeT node
class TreeData(object):
    def __init__(self, height=0, lids=[], miss_side='', comb_side='', word =''):
        self.height = height
        self.leaves = lids
        self.miss_side = miss_side
        self.comb_side = comb_side
        self.word = word

class TreeT(object):
    """Class that converts:
     PTB string <-> treelib <-> augmented tags
     """

    def __init__(self, max_id=0):
        self.tree = Tree()

    def from_ptb_to_tree(self, line, max_id=0, leaf_id=1, parent_id=None):
        # starts by ['(', 'pos']
        pos_tag = line[1]
        if parent_id is None:
            pos_id = 0
        else:
            pos_id = max_id
            max_id += 1

        self.tree.create_node(pos_tag, pos_id, parent_id, TreeData())

        parent_id = pos_id
        total_offset = 2

        if line[2] != '(':
            # sub-tree is leaf
            # line[0:3] = ['(', 'pos', 'word', ')']
            word_tag = line[2]
            self.tree.create_node(word_tag, leaf_id, parent_id, TreeData())
            return 4, max_id, leaf_id+1

        line = line[2:]

        while line[0] != ')':
            offset, max_id, leaf_id = self.from_ptb_to_tree(line, max_id, leaf_id, parent_id)
            total_offset += offset
            line = line[offset:]

        return total_offset+1, max_id, leaf_id

    def tree_to_path(self, nid, path):
        # Stop condition
        if self.tree[nid].is_leaf():
            path[nid] = []
            return nid, self.tree[nid].data.height

        # Recursion
        flag = CR
        for child in self.tree.children(nid):
            cid = child.identifier
            leaf_id, height = self.tree_to_path(cid, path)

            if (height == 0):
                # Reached end of path can add flag
                path[leaf_id].insert(0, flag)
                # path[leaf_id].append(flag)

            if height > 0:
                path[leaf_id].insert(0, nid)
                # only single child will have height>0
                # and its value will be the one that is returned
                # to the parent
                ret_leaf_id, ret_height = leaf_id, height-1

                # once we reached a height>0, it means that
                # this path includes the parent, and thus flag
                # direction should flip
                flag = CL

        return ret_leaf_id, ret_height

    def path_to_tags(self, path):
        tags = []
        for p in path:
            _res = []
            _p = copy.copy(p)
            if _p[0] in [CL, CR]:
                _res.append(_p[0])
                _p = _p[1:]
            while _p[:-1]:
                el_p = _p.pop(0)
                _res.append(self.tree[el_p].tag)
                for c in self.tree.children(el_p):
                    if c.identifier != _p[0]:
                        _res.append(R+c.tag if c.identifier > _p[0] else L+c.tag)
            _res.append(self.tree[_p[0]].tag)
            tags.append(_res)
        return tags

    def add_height(self, tree_dep):

        for n in self.tree.all_nodes():
            n.data.leaves = []

        for leaf in self.tree.leaves():
            lid = leaf.identifier
            hid = tree_dep[lid]
            if hid == self.tree.root:
                self.tree[lid].data.height = self.tree.depth(self.tree[lid])
                for cid in [p for p in self.tree.paths_to_leaves() if lid in p][0]:
                    self.tree[cid].data.leaves += [lid]
            else:
                height = -1
                cid = lid
                cond = True
                while cond:
                    self.tree[cid].data.leaves += [lid]
                    height += 1
                    cid = self.tree.parent(cid).identifier
                    cid_leaves = [l.identifier for l in self.tree.leaves(cid)]
                    cid_l_dep = [tree_dep[l] for l in cid_leaves if l != lid]
                    cond = set(cid_l_dep).issubset(set(cid_leaves))
                self.tree[lid].data.height = height

        x_nodes = [n.identifier for n in self.tree.all_nodes() if n.data.leaves == []]
        for x_node in x_nodes[::-1]:
            min_id = min(self.tree.children(x_node), key=lambda c: c.data.height)
            _lid = min_id.data.leaves[0]
            self.tree[_lid].data.height += 1
            self.tree[x_node].data.leaves += [_lid]

        return True

    def from_ptb_to_tag(self, line, max_id, depend):
        path = {}
        self.from_ptb_to_tree(line, max_id)
        self.add_height(depend)
        self.tree_to_path(self.tree.root, path)
        return self.path_to_tags(path.values())
