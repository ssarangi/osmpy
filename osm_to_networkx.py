"""
Read graphs in Open Street Maps osm format
Based on osm.py from brianw's osmgeocode
http://github.com/brianw/osmgeocode, which is based on osm.py from
comes from Graphserver:
http://github.com/bmander/graphserver/tree/master and is copyright (c)
2007, Brandon Martin-Anderson under the BSD License
"""

import xml.sax
import copy
import networkx as nx
import sys
from matplotlib.lines import Line2D
from matplotlib.widgets import Cursor, Button, CheckButtons
import numpy as np
import math
import shortest_path
import pylab as plt
# import vispy.mpl_plot as plt
# from vispy_renderer import *
# from vispy import app

class Node:
    def __init__(self, id, lon, lat):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.tags = {}

    def __str__(self):
        return str(self.id) # + " --> " + '(' + str(self.lat) + ',' + str(self.lon) + ')'

    __repr__ = __str__

class Way:
    def __init__(self, id, osm):
        self.osm = osm
        self.id = id
        self.nds = []
        self.tags = {}

    def split(self, dividers):
        # slice the node-array using this nifty recursive function
        def slice_array(ar, dividers):
            for i in range(1,len(ar)-1):
                if dividers[ar[i]]>1:
                    #print "slice at %s"%ar[i]
                    left = ar[:i+1]
                    right = ar[i:]

                    rightsliced = slice_array(right, dividers)

                    return [left]+rightsliced
            return [ar]



        slices = slice_array(self.nds, dividers)

        # create a way object for each node-array slice
        ret = []
        i=0
        for slice in slices:
            littleway = copy.copy( self )
            littleway.id += "-%d"%i
            littleway.nds = slice
            ret.append( littleway )
            i += 1

        return ret

class OSM:
    def __init__(self, filename_or_stream):
        """ File can be either a filename or stream/file object."""
        nodes = {}
        ways = {}
        bounds = {}

        superself = self

        class OSMHandler(xml.sax.ContentHandler):
            @classmethod
            def setDocumentLocator(self,loc):
                pass

            @classmethod
            def startDocument(self):
                pass

            @classmethod
            def endDocument(self):
                pass

            @classmethod
            def startElement(self, name, attrs):
                if name=='node':
                    self.currElem = Node(attrs['id'], float(attrs['lon']), float(attrs['lat']))
                elif name=='way':
                    self.currElem = Way(attrs['id'], superself)
                elif name=='tag':
                    self.currElem.tags[attrs['k']] = attrs['v']
                elif name=='nd':
                    self.currElem.nds.append( attrs['ref'] )
                elif name == 'bounds':
                    bounds['maxlon'] = float(attrs['maxlon'])
                    bounds['minlon'] = float(attrs['minlon'])
                    bounds['maxlat'] = float(attrs['maxlat'])
                    bounds['minlat'] = float(attrs['minlat'])

            @classmethod
            def endElement(self,name):
                if name=='node':
                    nodes[self.currElem.id] = self.currElem
                elif name=='way':
                    ways[self.currElem.id] = self.currElem

            @classmethod
            def characters(self, chars):
                pass

        xml.sax.parse(filename_or_stream, OSMHandler)
        self.nodes = nodes
        self.ways = ways
        self.bounds = bounds

        #"""
        #count times each node is used
        node_histogram = dict.fromkeys( list(self.nodes.keys()), 0 )
        for way in list(self.ways.values()):
            if len(way.nds) < 2:       #if a way has only one node, delete it out of the osm collection
                del self.ways[way.id]
            else:
                for node in way.nds:
                    node_histogram[node] += 1

        #use that histogram to split all ways, replacing the member set of ways
        new_ways = {}
        for id, way in self.ways.items():
            split_ways = way.split(node_histogram)
            for split_way in split_ways:
                new_ways[split_way.id] = split_way
        self.ways = new_ways


def read_osm(filename_or_stream, only_roads=True):
    osm = OSM(filename_or_stream)
    G = nx.Graph()

    for w in osm.ways.values():
        if only_roads and 'highway' not in w.tags:
            continue

        actual_nodes = w.nds
        nds = actual_nodes
        nds1 = actual_nodes[1:]
        weights = [shortest_path.calc_distance(osm.nodes[edge[0]], osm.nodes[edge[1]]) for edge in zip(nds, nds1)]

        for i, edge in enumerate(zip(nds, nds1)):
            G.add_edge(edge[0], edge[1], weight=weights[i])

        # G.add_path(w.nds, id=w.id, highway = w.tags['highway'], weight=weights)#{str(k): type(v) for k,v in w.tags.items()})
        #
        # if 'oneway' not in w.tags and  w.tags['highway'] != 'motorway':
        #     G.add_path(reversed(w.nds), id=w.id, highway = w.tags['highway'], weights=reversed(weights))
        #
        # elif w.tags['oneway'] != 'yes' and w.tags['oneway'] != '-1' and  w.tags['highway'] != 'motorway':
        #     G.add_path(reversed(w.nds), id=w.id, highway = w.tags['highway'], weights=reversed(weights))

    # for n_id in G.nodes_iter():
    #     n = osm.nodes[n_id]
    #     G.node[n_id] = dict(lon=n.lon,lat=n.lat)
    return G, osm

class MatplotLibMap:
    renderingRules = {
        'primary': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           ='#ee82ee',
                zorder          = -1,
        ),
        'primary_link': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = '#da70d6',
                zorder          = -1,
        ),
        'secondary': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = '#d8bfd8',
                zorder          = -2,
        ),
        'secondary_link': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = '#d8bfd8',
                zorder          = -2,
        ),
        'tertiary': dict(
                linestyle       = '-',
                linewidth       = 4,
                color           = (0.0,0.0,0.7),
                zorder          = -3,
        ),
        'tertiary_link': dict(
                linestyle       = '-',
                linewidth       = 4,
                color           = (0.0,0.0,0.7),
                zorder          = -3,
        ),
        'residential': dict(
                linestyle       = '-',
                linewidth       = 1,
                color           = (0.1,0.1,0.1),
                zorder          = -99,
        ),
        'unclassified': dict(
                linestyle       = ':',
                linewidth       = 1,
                color           = (0.5,0.5,0.5),
                zorder          = -1,
        ),
        'calculated_path': dict(
                linestyle       = '-',
                linewidth       = 4,
                color           = (1.0,0.0,0.0),
                zorder          = 1,
        ),
        'correct_path': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = (0.6,0.8,0.0),
                zorder          = 2,
        ),
        'default': dict(
                linestyle       = '-',
                linewidth       = 3,
                color           = 'b',
                zorder          = -1,
                ),
        }


    def __init__(self, osm, graph):
        self._node1 = None
        self._node2 = None
        self._mouse_click1 = None
        self._mouse_click2 = None
        self._node_map = {}
        self._graph = graph
        self._osm = None
        self._fig = None

        # Matplotlib data members
        self._node_plots = []
        self._osm = osm
        #list of lats and longs
        self.l_coordinates = []

        self.setup_figure()

    @property
    def node1(self):
        return self._node1

    @property
    def node2(self):
        return self._node2

    @property
    def coordinates(self):
        return self.l_coordinates


    def setup_figure(self):
        # get bounds from OSM data
        minX = float(self._osm.bounds['minlon'])
        maxX = float(self._osm.bounds['maxlon'])
        minY = float(self._osm.bounds['minlat'])
        maxY = float(self._osm.bounds['maxlat'])

        if self._fig is not None:
            plt.close(self._fig)

        self._fig = plt.figure(figsize=(20, 12), dpi=80, facecolor='grey', edgecolor='k')

        self._render_axes0 = self._fig.add_subplot(221, autoscale_on = False, xlim = (minX,maxX), ylim = (minY,maxY))
        self._render_axes0.xaxis.set_visible(False)
        self._render_axes0.yaxis.set_visible(False)
        plt.title("Dijkstra")

        self._render_axes1 = self._fig.add_subplot(222, autoscale_on = False, xlim = (minX,maxX), ylim = (minY,maxY))
        self._render_axes1.xaxis.set_visible(False)
        self._render_axes1.yaxis.set_visible(False)
        plt.title("A Star")

        self._render_axes2 = self._fig.add_subplot(223, autoscale_on = False, xlim = (minX,maxX), ylim = (minY,maxY))
        self._render_axes2.xaxis.set_visible(False)
        self._render_axes2.yaxis.set_visible(False)
        plt.title("Dijkstra Paths Considered")

        self._render_axes3 = self._fig.add_subplot(224, autoscale_on = False, xlim = (minX,maxX), ylim = (minY,maxY))
        self._render_axes3.xaxis.set_visible(False)
        self._render_axes3.yaxis.set_visible(False)
        plt.title("A Star Paths Considered")

        self.render(self._render_axes0)
        self.render(self._render_axes1)

        self._axes = {}
        self._axes['dijkstra'] = {}
        self._axes['astar'] = {}
        self._axes['dijkstra']['main'] = self._render_axes0
        self._axes['dijkstra']['paths_considered'] = self._render_axes2

        self._axes['astar']['main'] = self._render_axes1
        self._axes['astar']['paths_considered'] = self._render_axes3
        plt.show()

    def _get_axes(self, algo, graph_type):
        return self._axes[algo][graph_type]

    def render(self, axes, plot_nodes=False):
        plt.sca(axes)
        for idx, nodeID in enumerate(self._osm.ways.keys()):
            wayTags = self._osm.ways[nodeID].tags
            wayType = None
            if 'highway' in wayTags.keys():
                wayType = wayTags['highway']

            if wayType in [
                           'primary',
                           'primary_link',
                           'unclassified',
                           'secondary',
                           'secondary_link',
                           'tertiary',
                           'tertiary_link',
                           'residential',
                           'trunk',
                           'trunk_link',
                           'motorway',
                           'motorway_link'
                            ]:
                oldX = None
                oldY = None

                if wayType in list(MatplotLibMap.renderingRules.keys()):
                    thisRendering = MatplotLibMap.renderingRules[wayType]
                else:
                    thisRendering = MatplotLibMap.renderingRules['default']

                for nCnt, nID in enumerate(self._osm.ways[nodeID].nds):
                    y = float(self._osm.nodes[nID].lat)
                    x = float(self._osm.nodes[nID].lon)

                    self._node_map[(x, y)] = nID

                    if oldX is None:
                        pass
                    else:
                        plt.plot([oldX,x],[oldY,y],
                                marker          = '',
                                linestyle       = thisRendering['linestyle'],
                                linewidth       = thisRendering['linewidth'],
                                color           = thisRendering['color'],
                                solid_capstyle  = 'round',
                                solid_joinstyle = 'round',
                                zorder          = thisRendering['zorder'],
                                picker=2
                        )

                        if plot_nodes == True and (nCnt == 0 or nCnt == len(self._osm.ways[nodeID].nds) - 1):
                            plt.plot(x, y,'ro', zorder=5)

                    oldX = x
                    oldY = y

        self._fig.canvas.mpl_connect('pick_event', self.__onclick__)
        plt.draw()
        self.l_coordinates.append([x,y])

    def __clear_button_clicked__(self, event):
        print("Right Click")
        self._node1 = None
        self._node2 = None
        self._mouse_click1 = None
        self._mouse_click2 = None
        self.render(self._osm, plot_nodes=False)

    def __onclick__(self, event):
        threshold = 0.001

        if self._node1 is not None and self._node2 is not None:
            return None

        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            point = (float(np.take(xdata, ind)[0]), float(np.take(ydata, ind)[0]))
            node_id = self._node_map[point]

            if self._node1 is None:
                self._node1 = Node(node_id, point[0], point[1])
                self._mouse_click1 = (event.mouseevent.xdata, event.mouseevent.ydata)
                plt.sca(self._get_axes('dijkstra', 'main'))
                plt.plot(self._mouse_click1[0], self._mouse_click1[1], 'bo', zorder=10)
                plt.sca(self._get_axes('astar', 'main'))
                plt.plot(self._mouse_click1[0], self._mouse_click1[1], 'bo', zorder=10)
                plt.draw()
                return self._node1
            else:
                # Do not allow clicking of node id's within 100 node distances
                if abs(point[0] - self._node1.lon) < threshold and abs(point[1] - self._node1.lat) < threshold:
                    return None

                self._node2 = Node(node_id, point[0], point[1])
                self._mouse_click2 = (event.mouseevent.xdata, event.mouseevent.ydata)
                print("Both points marked")

                plt.sca(self._get_axes('dijkstra', 'main'))
                plt.plot(self._mouse_click2[0], self._mouse_click2[1], 'bo', zorder=10)
                plt.sca(self._get_axes('astar', 'main'))
                plt.plot(self._mouse_click2[0], self._mouse_click2[1], 'bo', zorder=10)
                plt.draw()

                # Now both the points have been marked. Now try to find a path.
                path_dijkstra, paths_considered_dijkstra = shortest_path.dijkstra(self._graph, self._node1.id, self._node2.id)
                path_astar, paths_considered_astar = shortest_path.astar(self._graph, self._node1.id, self._node2.id, self._osm)

                self.plot_path(self._get_axes('dijkstra', 'main'), path_dijkstra, MatplotLibMap.renderingRules['correct_path'], animate=False)
                self.plot_considered_paths(self._get_axes('dijkstra', 'paths_considered'), path_dijkstra, paths_considered_dijkstra)

                self.plot_path(self._get_axes('astar', 'main'), path_astar, MatplotLibMap.renderingRules['correct_path'], animate=False)
                self.plot_considered_paths(self._get_axes('astar', 'paths_considered'), path_astar, paths_considered_astar)

                return self._node2

    def plot_path(self, axes, path, rendering_style=None, animate=False):
        plt.sca(axes)
        edges = zip(path, path[1:])

        if rendering_style is None:
            thisRendering = MatplotLibMap.renderingRules['calculated_path']
        else:
            thisRendering = rendering_style

        for i, edge in enumerate(edges):
            node_from = self._osm.nodes[edge[0]]
            node_to = self._osm.nodes[edge[1]]
            x_from = node_from.lon
            y_from = node_from.lat
            x_to = node_to.lon
            y_to = node_to.lat

            if i == 0:
                x_from = self._mouse_click1[0]
                y_from = self._mouse_click1[1]

            if i == len(path) - 2:
                x_to = self._mouse_click2[0]
                y_to = self._mouse_click2[1]

            plt.plot([x_from,x_to],[y_from,y_to],
                    marker          = '',
                    linestyle       = thisRendering['linestyle'],
                    linewidth       = thisRendering['linewidth'],
                    color           = thisRendering['color'],
                    solid_capstyle  = 'round',
                    solid_joinstyle = 'round',
                    zorder          = thisRendering['zorder'],
                    )

            if animate:
                plt.draw()

        plt.draw()

    def plot_considered_paths(self, axes, path, paths_considered):
        plt.sca(axes)
        edges = zip(path, path[1:])

        # Render all the paths considered
        for i, edge in enumerate(paths_considered):
            node_from = self._osm.nodes[edge[0]]
            node_to = self._osm.nodes[edge[1]]
            x_from = node_from.lon
            y_from = node_from.lat
            x_to = node_to.lon
            y_to = node_to.lat

            plt.plot([x_from,x_to],[y_from,y_to],
                    marker          = '',
                    linestyle       = '-',
                    linewidth       = 1,
                    color           = 'green',
                    solid_capstyle  = 'round',
                    solid_joinstyle = 'round',
                    zorder          = 0,
                    )

        # Render all the paths considered
        for i, edge in enumerate(edges):
            node_from = self._osm.nodes[edge[0]]
            node_to = self._osm.nodes[edge[1]]
            x_from = node_from.lon
            y_from = node_from.lat
            x_to = node_to.lon
            y_to = node_to.lat

            if i == 0:
                x_from = self._mouse_click1[0]
                y_from = self._mouse_click1[1]

            if i == len(path) - 2:
                x_to = self._mouse_click2[0]
                y_to = self._mouse_click2[1]

            plt.plot([x_from,x_to],[y_from,y_to],
                    marker          = '',
                    linestyle       = '-',
                    linewidth       = 3,
                    color           = 'black',
                    solid_capstyle  = 'round',
                    solid_joinstyle = 'round',
                    zorder          = 1,
                    )

        plt.draw()


def convert_to_pixel_coords(lat, lon):
    map_width = 400
    map_height = 400

    x = (lon + 180.0) * (map_width / 360.0)
    lat_rad = abs(lat * math.pi / 180.0)

    merc_n = math.log(math.tan((math.pi / 4.0) + (lat_rad / 2.0)))

    y = 180.0/math.pi*math.log(math.tan(math.pi/4.0 + lat *(math.pi/180.0)/2.0))
    return x, y


class MapInfo:
    def __init__(self):
        self.map_shiftX = 0
        self.map_shiftY = 0


    def convert_lat_to_y(self, lat):
        y = 0
        w = 2000
        SCALE = 9000
        lat_rad = math.radians(lat)
        y = (w / (2 * math.pi) * math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0)) * SCALE)
        y += self.map_shiftY
        return y

    def convert_lon_to_x(self, lon):
        x = 0
        w = 2000
        SCALE = 9000
        lon_rad = math.radians(lon)

        x = (w / (2.0 * math.pi)) * (lon_rad) * SCALE
        x -= self.map_shiftX

        return x

def get_points_from_node_ids(osm, path):
    edges = zip(path, path[1:])

    path_coords = []

    for i, edge in enumerate(edges):
        node_from = osm.nodes[edge[0]]
        node_to = osm.nodes[edge[1]]
        x_from = node_from.lon
        y_from = node_from.lat
        x_to = node_to.lon
        y_to = node_to.lat

        x_pixel, y_pixel = convert_to_pixel_coords(y_from, x_from)
        path_coords.append([x_pixel, y_pixel, 0.1])

        if i == len(path) - 2:
            x_pixel, y_pixel = convert_to_pixel_coords(y_to, x_to)
            path_coords.append([x_pixel, y_pixel, 0.1])

    return np.array(path_coords).astype(np.float32)

def main():
    graph, osm = read_osm(sys.argv[1])
    print(osm.bounds)
    matplotmap = MatplotLibMap(osm, graph)
    # path = shortest_path.dijkstra(graph, '1081079917', '65501510')
    # points = get_points_from_node_ids(osm, path)
    # c = Canvas(points)
    # app.run()

if __name__ == "__main__":
    main()