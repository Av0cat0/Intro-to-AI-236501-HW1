import numpy as np
from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict
from collections import deque

PORTAL_COST = 100


class Node:
    def __init__(self, state, parent=None, action=0, cost=0, terminated=False, g=0, h=0, f=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.terminated = terminated
        self.g = g
        self.h = h
        self.f = f

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        return self.state < other.state

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def print_node(self):
        print(f"state: {self.state}")
        print(f"f: {self.f}")



# General agent to be inherited from
class Agent:
    def __init__(self):
        # Open variable will be changed on further algorithms,
        # If we'll make it we'll update the code to be implemented better
        self.open = deque()
        self.env = None
        self.close = set()
        self.expanded_nodes = 0
        self.total_cost = 0
        self.final_actions = []

    # Updating env as we don't accept the env variable at initiation
    def update_env(self, env: CampusEnv):
        self.env = env
        self.env.reset()

    # Expand node to get its children, returns a list of children
    def expand(self, node):
        c = []
        self.expanded_nodes += 1
        if node.terminated is False:
            for action, (state, cost, terminated) in self.env.succ(node.state).items():
                # We add to node_dst the action that caused us to get from node_src to node_dst
                c.append(Node(state, parent=node, action=action, cost=cost, terminated=terminated))
        return c

    # Back track the path to the current node
    # and then reverse the action list so the first action will be the first step
    def find_path(self, node):
        while node.parent is not None:
            self.total_cost += node.cost
            self.final_actions.append(node.action)
            node = node.parent
        return self.final_actions[::-1], self.total_cost, self.expanded_nodes

    def h_campus(self, state):
        # Convert state into position and extract based on its position the Manhattan distance
        row, col = self.env.to_row_col(state)
        pos = [self.env.to_row_col(goal) for goal in self.env.get_goal_states()]
        h = [abs(row - goal_row) + abs(col - goal_col) for goal_row, goal_col in pos]
        h.append(PORTAL_COST)
        return min(h)


class DFSGAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def DFS_recursion(self):
        curr_node = self.open.pop()
        if self.env.is_final_state(curr_node.state):
            return self.find_path(curr_node)
        self.close.add(curr_node.state)
        for child_node in self.expand(curr_node):
            curr_state = child_node.state
            open_states = [node.state for node in self.open]
            # The condition related to DFS-G rather than DFS
            if curr_state not in self.close.union(open_states):
                self.open.append(child_node)
                result = self.DFS_recursion()
                if result is not None:
                    return result
        # Reaching here meaning we're not able to progress in this path
        # And we haven't met our goal destination
        return None

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.__init__()
        self.update_env(env)
        # Appending root to open
        self.open.append(Node(env.get_initial_state()))
        return self.DFS_recursion()


class UCSAgent(Agent):

    def __init__(self) -> None:
        super().__init__()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.__init__()
        self.open = heapdict.heapdict()
        self.update_env(env)
        src_state = self.env.get_initial_state()
        node = Node(src_state)
        self.open[node.state] = (0,node.state,node)
        while len(self.open) > 0:
            curr_state, (path, state, curr_node) = self.open.popitem()
            if env.is_final_state(curr_state):
                return self.find_path(curr_node)
            self.close.add(curr_node)
            if curr_node.cost != np.inf:
                for child_node in self.expand(curr_node):
                    g = curr_node.g + child_node.cost
                    if child_node in self.close:
                        continue
                    if child_node.state not in self.open or g < self.open[child_node.state][2].g:
                        child_node.g = g
                        self.open[child_node.state] = (g, child_node.state,child_node)
        return [], float('inf'), self.expanded_nodes


class WeightedAStarAgent(Agent):

    def __init__(self):
        super().__init__()

    def f(self, node, w):
        if w == 1:
            return node.h
        elif w == 0:
            return node.g
        else:
            return ((1 - w) * node.g) + (w * node.h), 4

    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        self.__init__()
        self.open = heapdict.heapdict()
        self.close = heapdict.heapdict()
        self.update_env(env)
        src_state = self.env.get_initial_state()
        node = Node(src_state)
        node.h = self.h_campus(src_state)
        node.g = 0
        node.f = self.f(node, h_weight)
        self.open[node.state] = node
        while len(self.open) > 0:
            curr_state, curr_node = self.open.popitem()
            self.close[curr_node.state] = curr_node
            if env.is_final_state(curr_state):
                return self.find_path(curr_node)
            if curr_node.cost != np.inf:
                for child_node in self.expand(curr_node):
                    g = curr_node.g + child_node.cost
                    if child_node.state in self.close.keys():
                        if g < self.close[child_node.state].g:
                            new_node = Node(child_node.state, parent=curr_node, h=self.h_campus(child_node.state), g=g, f=self.f(child_node,h_weight),action=child_node.action, cost=child_node.cost)
                            self.close.pop(new_node.state)
                            self.open[child_node.state] = new_node
                        continue
                    if child_node.state not in self.open.keys() or g < self.open[child_node.state].g:
                        child_node.g = g
                        child_node.h = self.h_campus(child_node.state)
                        child_node.f = self.f(child_node, h_weight)
                        self.open[child_node.state] = child_node
        return [], float('inf'), self.expanded_nodes



class AStarAgent(WeightedAStarAgent):

    def __init__(self):
        super().__init__()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        return super().search(env, 0.5)

#%%
def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
    self.__init__()
    self.open = heapdict.heapdict()
    self.update_env(env)
    src_state = self.env.get_initial_state()
    node = Node(src_state)
    node.h = self.h_campus(src_state)
    node.g = 0
    node.f = self.f(node, h_weight)
    self.open[node.state] = node
    while len(self.open) > 0:
        curr_state, curr_node = self.open.popitem()
        print(f"Expanding node: {curr_state} with f={curr_node.f}, g={curr_node.g}, h={curr_node.h}")
        if env.is_final_state(curr_state):
            return self.find_path(curr_node)
        self.close.add(curr_node)
        if curr_node.cost != np.inf:
            for child_node in self.expand(curr_node):
                g = curr_node.g + child_node.cost
                if child_node in self.close:
                    continue
                if child_node.state not in self.open or g < self.open[child_node.state].g:
                    child_node.g = g
                    child_node.h = self.h_campus(child_node.state)
                    child_node.f = self.f(child_node, h_weight)
                    self.open[child_node.state] = child_node
    return [], float('inf'), self.expanded_nodes

#%%
