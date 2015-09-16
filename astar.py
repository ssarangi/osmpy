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
    def __init__(self):
        self.__heapq = []

    def push(self, item, priority = 0):
        self.__heapq.append((priority, item))
        heapq.heapify(self.__heapq)

    def pop(self):
        # item is the 1th index
        return heapq.heappop(self.__heapq)[1]

    def empty(self):
        return len(self.__heapq) == 0


def draw_graph(graph_edges):

    # create networkx graph
    G = nx.Graph()

    # add edges
    edge_weights = []
    for edge in graph_edges:
        edge_weight = random.randint(1, 10)
        edge_weights.append(edge_weight)
        G.add_edge(edge[0], edge[1], weights=edge_weight)

    return G, edge_weights

def render_graph(G, graph_edges, edge_weights, labels=None, graph_layout='shell',
                 node_size=1600, node_color='blue', node_alpha=0.3,
                 node_text_size=12,
                 edge_color='blue', edge_alpha=0.3, edge_tickness=1,
                 edge_text_pos=0.3,
                 text_font='sans-serif'):
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
        labels = range(len(graph_edges))

    edge_labels = dict(zip(graph_edges, edge_weights))
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels,
                                 label_pos=edge_text_pos)

    # show graph
    plt.show()
    return G

def dijkstra(graph, start, end, graph_edges, edge_weights):
    cost = [float('inf')] * len(graph.nodes())
    cost[start] = 0
    path = [-1] * len(graph.nodes())
    visited = []

    pq = PriorityQueue()
    pq.push(start, priority=0)
    while not pq.empty():
        node = pq.pop()
        visited.append(node)
        for neighbor in graph.neighbors(node):
            edge_weight = int(graph.get_edge_data(node, neighbor)['weights'])
            new_edge_weight = cost[node] + edge_weight

            if cost[neighbor] > new_edge_weight:
                cost[neighbor] = min(cost[neighbor], new_edge_weight)
                path[neighbor] = node

                if neighbor not in visited:
                    pq.push(neighbor, priority = int(cost[neighbor]))

    newpath = [end]
    while True:
        if end == start:
            break

        newpath.append(path[end])
        end = path[end]

    newpath.reverse()
    print(newpath)

    # Render the map
    # these are different layouts for the network you may try
    # shell seems to work best
    graph_pos = nx.spring_layout(graph)

    # Find the nodes which are not in the path
    nodes_not_in_path = list(set(graph.nodes()) - set(newpath))

    # draw graph
    nx.draw_networkx_nodes(graph, graph_pos, node_size=1600, nodelist = nodes_not_in_path,
                           alpha=0.7, node_color='blue')

    nx.draw_networkx_nodes(graph, graph_pos, node_size=1600, nodelist = newpath,
                           alpha=0.7, node_color='red')


    edges_in_path = [edge for edge in zip(newpath, newpath[1:])]

    edges_not_in_path = list(set(graph_edges) - set(edges_in_path))

    nx.draw_networkx_edges(graph, graph_pos, width=1, edgelist=edges_not_in_path,
                           alpha=0.5, edge_color='black')

    nx.draw_networkx_edges(graph, graph_pos, width=3, edgelist=edges_in_path,
                           alpha=1.0, edge_color='red')

    nx.draw_networkx_labels(graph, graph_pos, font_size=12,
                            font_family='sans-serif')


    edge_labels = dict(zip(graph_edges, edge_weights))
    nx.draw_networkx_edge_labels(graph, graph_pos, edge_labels=edge_labels,
                                 label_pos=0.3)

    # show graph
    plt.show()

    return newpath


def AStar(graph, start, end):
    return 0

def main():
    graph_edges = [(0, 1), (1, 5), (1, 7), (4, 5), (4, 8), (1, 6), (3, 7), (5, 9),
                   (2, 4), (0, 4), (2, 5), (3, 6), (8, 9)]

    # you may name your edge labels
    labels = map(chr, range(65, 65+len(graph_edges)))

    # if edge labels is not specified, numeric labels (0, 1, 2...) will be used
    graph, edge_weights = draw_graph(graph_edges)
    dijkstra(graph, 0, 2, graph_edges, edge_weights)
    AStar(graph, 0, 2)

if __name__ == "__main__":
    main()