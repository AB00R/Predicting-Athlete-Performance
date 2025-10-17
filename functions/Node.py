class Node:
    """
    A node in the search tree representing a state in the search space.
    
    Each node contains a state (day, fatigue, risk, performance, history),
    a reference to its parent node, the action that led to this state,
    and various cost metrics used by search algorithms.
    
    Attributes:
        state: Tuple containing (day, fatigue, risk, performance, history)
        parent: Reference to the parent Node
        action: The action (intensity, duration) that led to this state
        g: Path cost from start to this node
        f: Total evaluation function value (g + h)
        h: Heuristic value (estimated cost to goal)
        depth: Depth of this node in the search tree
    """
    def __init__(self, state, parent=None, action=None, g=0, f=0, h=0, costless=False):
        self.state = state
        self.parent = parent
        self.action = action
        if costless == False:
            self.g = g
            self.f = f
            self.h = h

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state
    
    def __gt__(self, other):
        return isinstance(other, Node) and self.f > other.f
