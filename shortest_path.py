__author__ = 'sarangis'

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.shortest_paths import *
import random

import heapq
from math import *

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


def calc_distance(node,otherNode):
    lat1 = float(node.lat)
    lon1 = float(node.lon)
    lat2 = float(otherNode.lat)
    lon2 = float(otherNode.lon)

    #Code coppied form: http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = list(map(radians, [lon1, lat1, lon2, lat2]))

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km

def draw_graph(graph_edges):

    # create networkx graph
    G = nx.Graph()

    # add edges
    edge_weights = []
    for edge in graph_edges:
        edge_weight = random.randint(1, 10)
        edge_weights.append(edge_weight)
        G.add_edge(edge[0], edge[1], weight=edge_weight)

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

def reconstruct_path(came_from, start, end):
    current = end
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def dijkstra(graph, start, end):
    pq = PriorityQueue()
    pq.push(start, priority=0)
    came_from = {}
    cost_so_far = {}
    cost_so_far[start] = 0
    came_from[start] = None
    paths_considered = []

    while not pq.empty():
        current = pq.pop()

        if current == end:
            break

        for neighbor in graph.neighbors(current):
            new_cost = cost_so_far[current] + float(graph.get_edge_data(current, neighbor)['weight'])
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                pq.push(neighbor, priority = new_cost)
                came_from[neighbor] = current
                paths_considered.append((current, neighbor))

    path = reconstruct_path(came_from, start, end)
    return path, paths_considered


def astar(graph, start, end, osm):
    pq = PriorityQueue()
    pq.push(start, priority=0)

    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    paths_considered = []

    while not pq.empty():
        current = pq.pop()

        if current == end:
            break

        for neighbor in graph.neighbors(current):
            new_cost = cost_so_far[current] + float(graph.get_edge_data(current, neighbor)['weight'])
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + calc_distance(osm.nodes[end], osm.nodes[neighbor])
                pq.push(neighbor, priority = priority)
                came_from[neighbor] = current
                paths_considered.append((current, neighbor))


    path = reconstruct_path(came_from, start, end)
    return path, paths_considered

def main():
    graph_edges = [(0, 1), (1, 5), (1, 7), (4, 5), (4, 8), (1, 6), (3, 7), (5, 9),
                   (2, 4), (0, 4), (2, 5), (3, 6), (8, 9)]

    # you may name your edge labels
    labels = map(chr, range(65, 65+len(graph_edges)))

    # if edge labels is not specified, numeric labels (0, 1, 2...) will be used
    graph, edge_weights = draw_graph(graph_edges)
    dijkstra(graph, 0, 2)
    # AStar(graph, 0, 2)

if __name__ == "__main__":
    main()