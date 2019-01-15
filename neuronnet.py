# neuronnet.py

import numpy as np

class Neuron:
    """A neuron class for neuronnets. Contains a value, a
    reference to the parent neurons, and references to a variable number of child neurons.

    Attributes:
    threshold (float): value at which neuron will fire.
    excitability (float): I dont think this helps for anything yet.
    value (float): value that neuron currently contains
    decay (float): ammount that neuron will 'decrease' between firings
    lag (float): may contain the ammount of timesteps that need to pass after reaching
                the threashold before firing.
    dentrite (dic): of form {reception strength (float between 0 and 1):[[prior neurons],[how many times respective prior neurons have successively fired]]}
    axon (list): conaining post neurons
    fired (Bool): True if the threshold is surpassed

    """
    def __init__(self, threshold = .5, decay = .2):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.threshold = threshold
        self.value = 0
        # self.excitability = value - threshold
        self.decay = decay
        self.lag = None # will contain the 'lag time' (If I deem it necessary)
        self.dendrite = dict()
        self.axon = []
        self.fired = False

    def value_summation(self):
        '''sums together the weights that have been fired, and if the current
           value surpasses the threshold, this neuron also fires'''
        for weight in self.dendrite:
            self.value += sum(weight for n in self.dendrite[0][weight] if n.fired)

    def fire(self):
        '''Sets the fired value to true if value is over threashold- also resets value'''
        if self.value >= self.threshold:
            self.fired = True
            self.value = 0 # this may need to be changed to = self.value - self.firecost

    def decay_vals(self):
        '''decreases value by the ammount of decay'''
        self.value = self.value - self.decay

    def add_dendrite(self,node,weight):
        '''Adds the given node as a to the dendrite dictionary under the given weight'''
        if weight in self.dendrite:
            self.dendrite[weight].append(node)
        else:
            self.dendrite[weight] = [node]


class NNet:
    """Neuron net data structure class.
    The ins references the first nodes in the structure, and the
    outs references the last layer of nodes (the output options).
    recentlit holds the neurons that were most recently lit above a
    threshold value, 'threshold'""
    """
    def __init__(self, in_count = 4, out_count = 4, t = .5):
        self.ins = [Neuron() for i in range(in_count)]
        self.outs = [Neuron() for i in range(out_count)]
        recentlit = []
        threshold = t

    def create_nodes(self):
        '''Creates new nodes and connections'''

    def evaluate_nodes(self):
        '''Evaluates the nodes'''

    def fire_nodes(self):


    def remove_nodes(self):

    def predict(self):
        ''' returns a dictionary of the out neuron id number, and their expected value'''
        return {i:self.outs[i].value for i in len(self.outs)}

    def connect(self):
        ''' this could be done recursively, starting at ins and ending each time at outs.
        it would need to be done 'breadth first' however, that way the 'neuralness' could be
        maintained.  This would not allow for loops.  That modification will need to be made.'''
        for i in len(ins):
            pass

    def train(self, indata, outdata):
        if len(indata) != len(self.ins):
            raise ValueError("indata not same size as ins length")
        if len(outdata) != len(self.outs):
            raise ValueError("outdata not same size as outs length")




    def insert(self, layer, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        def step(current):
            """Recursively step through the tree until the location where
            the data should go is found.  Returns the parent node that
            the child should attatch to.
            """
            if data == current.value:               # this value is in the tree
                raise ValueError("There is already a node in this BST with this value.")
            # if this node is a leaf, return it
            if current.left is None and current.right is None:
                return current
            if data < current.value:                # Recursively search left.
                if current.left is None:            # If you need to branch to the left
                    return current
                else:
                    return step(current.left)
            else:                                   # Recursively search right.
                if current.right is None:            # If you need to branch to the right
                    return current
                else:
                    return step(current.right)

        n = BSTNode(data)
        # if the tree is empty, assign the 'root' attribute to the new BSTNode
        # contianing the data
        if self.root is None:
            self.root = n
        # if tree is not empty, we create a new BSTNode and put it in the
        # correct location
        else:
            parent = step(self.root)
            n.prev = parent
            if n.value < parent.value:
                parent.left = n
            if n.value > parent.value:
                parent.right = n

        return
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        # 3 CASES: 1: target is a leaf,
        #          2: target has one child,
        #          3: target has 2 children.
        # We call the target 'trash'
        ### Note that if the desired data is unavailable, find()
        ### will raise a ValueError.
        trash = self.find(data)
        # 1: If the target is a leaf node
        if trash.left is None and trash.right is None:
            # if it's the root
            if trash is self.root:
                self.root = None
                return
            # if it's to the left of the parent
            if trash.prev.value > trash.value:
                trash.prev.left = None
            # if it's to the right of the parent
            else:
                trash.prev.right = None
            # return out of case 1
            return
        # 2: If the target has one child
        elif trash.left is None or trash.right is None:
            # start by getting a reference to the child
            if trash.left is None:
                child = trash.right
            if trash.right is None:
                child = trash.left
            # if the target's the root
            if trash is self.root:
                self.root = child
                self.root.prev = None
                # in this case, you are done early
                return
            # if the target is to the left of its parent
            if trash.prev.value > trash.value:
                trash.prev.left = child
                child.prev = trash.prev
            # if the target is to the right of its parent
            if trash.prev.value < trash.value:
                trash.prev.right = child
                child.prev = trash.prev
            return
            # return out of case 2
        # 3: If the target has two children
        else:
            # find the immediate predecessor
            node_of_interest = trash.left
            pre = 0
            while node_of_interest is not None:
                node_of_interest, pre = node_of_interest.right, node_of_interest
            val = pre.value
            # use remove on predecessor's value
            self.remove(val)
            # replace target's value with the old immediate predecessors
            trash.value = val
            # return out of case 3
            return
        raise NotImplementedError("Problem 3 Incomplete")

    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()
