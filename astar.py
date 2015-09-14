__author__ = 'sarangis'

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.shortest_paths import *
import math
import random

import heapq

class PriorityQueue(object):
    '''
    Priority Queue based on a heap, capable of inserting a new node with desired priority, updating the priority of
    an existing node and deleting an arbitrary node while keeping invariant
    '''
    def __init__(self, heap = []):
        '''
        If heap is not empty, make sure its heapified
        :param heap:
        :return:
        '''
        heapq.heapify(heap)
        self.heap = heap
        self.entry_finder = dict({i[-1]: i for i in heap})
        self.REMOVED = '<remove marker>'

    def insert(self, node, priority=0):
        '''
        Entry finder
        :param node:
        :param priority:
        :return:
        '''
        if node in self.entry_finder:
            self.delete(node)

        entry = [priority, node]
        self.entry_finder[node] = entry
        heapq.heapify(self.heap, entry)

    def delete(self, node):
        entry = self.entry_finder.pop(node)
        entry[-1] = self.REMOVED
        return entry[0]

    def pop(self):
        while self.heap:
            priority, node = heapq.heappop(self.heap)
            if node is not self.REMOVED:
                del self.entry_finder[node]
                return priority, node
        raise KeyError("pop from an empty priority queue")


def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=1600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # create networkx graph
    G=nx.Graph()

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
    visited = {start: 0}
    path = {}

    print(graph.get_edge_data(start, 1))

def astar(graph, start, end):
    pass

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