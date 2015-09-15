__author__ = 'sarangis'

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.shortest_paths import *
import random

import heapq


class PriorityQueue(object):
    """Priority queue based on heap, capable of inserting a new node with
    desired priority, updating the priority of an existing node and deleting
    an abitrary node while keeping invariant"""

    def __init__(self, heap=[]):
        """if 'heap' is not empty, make sure it's heapified"""

        heapq.heapify(heap)
        self.heap = heap
        self.entry_finder = dict({i[-1]: i for i in heap})
        self.REMOVED = '<remove_marker>'

    @property
    def empty(self):
        '''
        This is inefficient but just look through the list and see if everything is marked for removal
        :return:
        '''
        for i in self.heap:
            if not isinstance(i[1], str):
                return False

        return True

    def insert(self, node, priority=0):
        """'entry_finder' bookkeeps all valid entries, which are bonded in
        'heap'. Changing an entry in either leads to changes in both."""

        if node in self.entry_finder:
            self.delete(node)
        entry = [priority, node]
        self.entry_finder[node] = entry
        heapq.heappush(self.heap, entry)

    def delete(self, node):
        """Instead of breaking invariant by direct removal of an entry, mark
        the entry as "REMOVED" in 'heap' and remove it from 'entry_finder'.
        Logic in 'pop()' properly takes care of the deleted nodes."""

        entry = self.entry_finder.pop(node)
        entry[-1] = self.REMOVED
        return entry[0]

    def pop(self):
        """Any popped node marked by "REMOVED" does not return, the deleted
        nodes might be popped or still in heap, either case is fine."""

        while self.heap:
            priority, node = heapq.heappop(self.heap)
            if node is not self.REMOVED:
                del self.entry_finder[node]
                return priority, node
        raise KeyError('pop from an empty priority queue')

def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=1600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # create networkx graph
    G = nx.Graph()

    # add edges
    edge_weights = []
    for edge in graph:
        edge_weight = random.randint(0, 10)
        edge_weights.append(edge_weight)
        G.add_edge(edge[0], edge[1], weights=edge_weight)

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos = nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos = nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos = nx.random_layout(G)
    else:
        graph_pos = nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G, graph_pos, node_size=node_size,
                           alpha=node_alpha, node_color=node_color)

    nx.draw_networkx_edges(G, graph_pos, width=edge_tickness,
                           alpha=edge_alpha, edge_color=edge_color)

    nx.draw_networkx_labels(G, graph_pos, font_size=node_text_size,
                            font_family=text_font)

    if labels is None:
        labels = range(len(graph))

    edge_labels = dict(zip(graph, edge_weights))
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels,
                                 label_pos=edge_text_pos)

    # show graph
    plt.show()
    return G

def dijkstra(graph, start, end):
    cost = [float('inf')] * len(graph.nodes())
    cost[start] = 0
    path = [-1] * len(graph.nodes())
    visited = []

    pq = PriorityQueue()
    pq.insert(start, priority=0)
    while not pq.empty:
        top = pq.pop()
        node = top[1]
        visited.append(node)
        for neighbor in graph.neighbors(node):
            edge_weight = int(graph.get_edge_data(node, neighbor)['weights'])
            new_edge_weight = cost[node] + edge_weight

            if cost[neighbor] > new_edge_weight:
                cost[neighbor] = min(cost[neighbor], new_edge_weight)
                path[neighbor] = node
                pq.insert(neighbor, priority=cost[neighbor])

    newpath = [end]
    while True:
        if end == start:
            break

        newpath.append(path[end])
        end = path[end]

    newpath.reverse()
    print(newpath)
    return newpath

def astar(graph, start, end):
    return 0

def main():
    graph_edges = [(0, 1), (1, 5), (1, 7), (4, 5), (4, 8), (1, 6), (3, 7), (5, 9),
                   (2, 4), (0, 4), (2, 5), (3, 6), (8, 9)]

    # you may name your edge labels
    labels = map(chr, range(65, 65+len(graph_edges)))

    # if edge labels is not specified, numeric labels (0, 1, 2...) will be used
    graph = draw_graph(graph_edges)
    dijkstra(graph, 0, 2)
    astar(graph, 0, 2)

if __name__ == "__main__":
    main()