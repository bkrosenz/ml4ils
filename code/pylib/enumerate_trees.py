# A very simple representation for Nodes. Leaves are anything which is not a Node.
class Node(object):
  def __init__(self, left, right):
    self.left = left
    self.right = right

  def __repr__(self):
    return '(%s, %s)' % (self.left, self.right)

# Given a tree and a label, yields every possible augmentation of the tree by
# adding a new node with the label as a child "above" some existing Node or Leaf.
def add_leaf(tree, label):
  yield Node(label, tree)
  if isinstance(tree, Node):
    for left in add_leaf(tree.left, label):
      yield Node(left, tree.right)
    for right in add_leaf(tree.right, label):
      yield Node(tree.left, right)

# Given a list of labels, yield each rooted, unordered full binary tree with
# the specified labels.
def enum_unordered(labels):
  if len(labels) == 1:
    yield labels[0]
  else:
    for tree in enum_unordered(labels[1:]):
      for new_tree in add_leaf(tree, labels[0]):
        yield new_tree

def enum_ordered(labels):
  if len(labels) == 1:
    yield labels[0]
  else:
    for i in range(1, len(labels)):
      for left in enum_ordered(labels[:i]):
        for right in enum_ordered(labels[i:]):
          yield Node(left, right)

## https://stackoverflow.com/questions/14900693/enumerate-all-full-labeled-binary-tree?lq=1
