import collections.abc

R = '}'
L = '{'
CR = '>'
CL = '<'
ANY = '*'

class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

        for child in self.children:
            child.parent = self

        self.parent = None

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, dependancy, index=0):
        tree = self
        children = []
        for child in tree.children:
            children.append(child.convert(dependancy, index=index))
            index = children[-1].right

        return InternalMyParseNode(tree.label, children)


class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

        self.parent = None

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, dependancy, index=0):
        return LeafMyParseNode(index, self.tag, self.word)(dependancy[index] - 1)


class MyParseNode(object):
    pass

class InternalMyParseNode(MyParseNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, MyParseNode) for child in children)
        assert children
        # assert all(
        #     left.right == right.left
        #     for left, right in zip(children, children[1:])
        #     if not (isinstance(left, MissMyParseNode)
        #             and isinstance(right, MissMyParseNode)))
        self.children = tuple(children)

        for child in self.children:
            child.parent = self

        self.left = children[0].left
        self.right = children[-1].right

        self.parent = None

    def __call__(self, keep_valence_value):
        self.serialize(keep_valence_value)
        return self

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def missing_leaves(self):
        for child in self.children:
            yield from child.missing_leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label, children)
        return tree

    def serialize(self, keep_valence_value):

        def helper(current, sibling):
            side = L if current.left > sibling.left else R
            if not keep_valence_value:
                return side+ANY
            return side+sibling.bracket_label()

        # Recursion
        flag = CR
        for child in self.children:
            winner_child_leaf = child.serialize(keep_valence_value)

            # Reached end of path can add flag
            if winner_child_leaf.dependancy in range(self.left, self.right) or (flag == CL):
                winner_child_leaf.labels.append(flag)
            else:
                # only single child will move to parent
                # and its value will be the one that is returned
                # to the parent
                winner_child_leaf.labels.append(self.label)
                winner_child_leaf.labels.extend([helper(child, sibling) for sibling in child.siblings()])
                ret_leaf_node = winner_child_leaf

                # once we reached here, it means that
                # this path includes the parent and thus flag
                # direction should flip
                flag = CL

        return ret_leaf_node

    def siblings(self):
        assert self.parent is not None

        for child in self.parent.children:
            if child != self:
                yield child
        # return [child for child in self.parent.children if child != self]

    def bracket_label(self):
        return self.label

    def combine(self, node_to_merge, node_to_remove):
        tree = self
        children = []
        for child in tree.children:
            children.append(child.combine(node_to_merge, node_to_remove))
        return InternalMyParseNode(tree.label, children)

class LeafMyParseNode(MyParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def __call__(self, dependency):
        assert isinstance(dependency, int)
        self.dependancy = dependency
        return self

    def bracket_label(self):
        return self.tag

    def leaves(self):
        yield self

    def missing_leaves(self):
        yield from ()

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

    def combine(self, node_to_merge, node_to_remove):
        return LeafMyParseNode(self.left, self.tag, self.word)

    def siblings(self):
        assert self.parent is not None

        for child in self.parent.children:
            if child != self:
                yield child
        # return [child for child in self.parent.children if child != self]

    def serialize(self, keep_valence_value):
        # self.label = tuple(self.tag)
        self.labels = []
        return self

    def deserialize(self, labels):
        # invalid labels: labels should start with token not missing
        if (labels[0].startswith(R) or labels[0].startswith(L)):
            return None
        index = self.left
        children = [self]
        while labels:
            p_label = labels[0]
            labels = labels[1:]
            while labels and (labels[0].startswith(R) or labels[0].startswith(L)):
                if labels[0].startswith(R):
                    index = children[-1].right
                    children += [MissMyParseNode(labels[0], index)]
                else:
                    index = children[0].left - 1
                    children = [MissMyParseNode(labels[0], index)] + children
                labels = labels[1:]
            children = [InternalMyParseNode(p_label, children)]
        return children[-1]

class MissMyParseNode(MyParseNode):
    def __init__(self, label, index = 0):
        self.label = label
        self.left = index
        self.right = index + 1

    def leaves(self):
        yield self

    def missing_leaves(self):
        yield self

    def siblings(self):
        assert self.parent is not None

        for child in self.parent.children:
            if child != self:
                yield child
        # return [child for child in self.parent.children if child != self]

    def convert(self):
        return LeafTreebankNode(self.label, self.label)

    def bracket_label(self):
        return self.label

    def combine(self, node_to_merge, node_to_remove):
        tree = self
        if tree == node_to_remove:
            return node_to_merge.combine(node_to_merge, node_to_remove)
        return MissMyParseNode(tree.label, tree.left)

def load_trees(path, strip_top=True):
    with open(path) as infile:
        tokens = infile.read().replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":

            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1
        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label == "TOP":
                assert len(tree.children) == 1
                trees[i] = tree.children[0]
                trees[i].parent = None

    return trees
