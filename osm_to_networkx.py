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
from math import *
import shortest_path
import pylab as plt

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


    def __init__(self, graph):
        self._node1 = None
        self._node2 = None
        self._mouse_click1 = None
        self._mouse_click2 = None
        self._node_map = {}
        self._graph = graph
        self._osm = None
        self._fig = None
        self._render_axes = None

        # Matplotlib data members
        self._node_plots = []

    @property
    def node1(self):
        return self._node1

    @property
    def node2(self):
        return self._node2

    def render(self, osm, plot_nodes=False, new_plot = True):
        self._osm = osm

        # get bounds from OSM data
        minX = float(osm.bounds['minlon'])
        maxX = float(osm.bounds['maxlon'])
        minY = float(osm.bounds['minlat'])
        maxY = float(osm.bounds['maxlat'])

        if new_plot:
            self._fig = plt.figure()
            self._render_axes = self._fig.add_subplot(111,autoscale_on=False,xlim=(minX,maxX),ylim=(minY,maxY))
            self._render_axes.xaxis.set_visible(False)
            self._render_axes.yaxis.set_visible(False)
            plt.subplots_adjust(bottom=0.2)
            axcheckbox = plt.axes([0.13, 0.05, 0.25, 0.075])
            axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
            clear_button = Button(axnext, 'Clear')
            clear_button.on_clicked(self.__clear_button_clicked__)
            animate_button = CheckButtons(axcheckbox, ('Show Nodes', 'Animate'), (False, False))
            plt.subplot(111, autoscale_on=False,xlim=(minX,maxX),ylim=(minY,maxY))
            # check_buttons = CheckButtons(ax, ('Show Nodes', 'Animate'), (False, False))
        else:
            plt.subplot(111, autoscale_on=False,xlim=(minX,maxX),ylim=(minY,maxY))
            plt.cla()
            plt.subplot(111, autoscale_on=False,xlim=(minX,maxX),ylim=(minY,maxY))

        for idx, nodeID in enumerate(osm.ways.keys()):
            wayTags = osm.ways[nodeID].tags
            wayType = None
            if 'highway' in wayTags.keys():
                wayType = wayTags['highway']

            if wayType in ['primary',
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
                           'motorway_link']:
                oldX = None
                oldY = None

                if wayType in list(MatplotLibMap.renderingRules.keys()):
                    thisRendering = MatplotLibMap.renderingRules[wayType]
                else:
                    thisRendering = MatplotLibMap.renderingRules['default']

                for nCnt, nID in enumerate(osm.ways[nodeID].nds):
                    y = float(osm.nodes[nID].lat)
                    x = float(osm.nodes[nID].lon)

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

                        if plot_nodes == True:
                            plt.plot(x, y,'ro', zorder=5)

                    oldX = x
                    oldY = y

        if new_plot:
            # but_ax=plt.subplot2grid((8,4),(7,0),colspan=1)
            # reset_button=Button(but_ax,'Reset')
            self._fig.canvas.mpl_connect('pick_event', self.__onclick__)
            plt.show()

    def __clear_button_clicked__(self, event):
        print("Right Click")
        self._node1 = None
        self._node2 = None
        self._mouse_click1 = None
        self._mouse_click2 = None
        self.render(self._osm, plot_nodes=False, new_plot=False)

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
                return self._node1
            else:
                # Do not allow clicking of node id's within 100 node distances
                if abs(point[0] - self._node1.lon) < threshold and abs(point[1] - self._node1.lat) < threshold:
                    return None

                self._node2 = Node(node_id, point[0], point[1])
                self._mouse_click2 = (event.mouseevent.xdata, event.mouseevent.ydata)
                print("Both points marked")

                # import simple_router
                # path = simple_router.run_simple_router(sys.argv[1])
                # self.plot_path(path, MatplotLibMap.renderingRules['correct_path'])

                plt.plot(self._mouse_click1[0], self._mouse_click1[1], 'bo', zorder=10)
                plt.plot(self._mouse_click2[0], self._mouse_click2[1], 'bo', zorder=10)
                plt.draw()

                # Now both the points have been marked. Now try to find a path.
                path = shortest_path.dijkstra(self._graph, self._node1.id, self._node2.id)
                # path = shortest_path.astar(self._graph, self._node1.id, self._node2.id, self._osm)
                self.plot_path(path, MatplotLibMap.renderingRules['correct_path'], animate=True)

                return self._node2

    def plot_path(self, path, rendering_style=None, animate=False):
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

def main():
    graph, osm = read_osm(sys.argv[1])
    print(osm.bounds)
    matplotmap = MatplotLibMap(graph)
    matplotmap.render(osm, plot_nodes=False)


if __name__ == "__main__":
    main()