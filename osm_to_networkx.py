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
import bokeh.plotting as plt

class Node:
    def __init__(self, id, lon, lat):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.tags = {}

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
    G = nx.DiGraph()

    for w in osm.ways.values():
        if only_roads and 'highway' not in w.tags:
            continue

        G.add_path(w.nds, id=w.id, highway = w.tags['highway'])#{str(k): type(v) for k,v in w.tags.items()})

        if 'oneway' not in w.tags and  w.tags['highway'] != 'motorway':
            G.add_path(reversed(w.nds), id=w.id, highway = w.tags['highway'])

        elif w.tags['oneway'] != 'yes' and w.tags['oneway'] != '-1' and  w.tags['highway'] != 'motorway':
            G.add_path(reversed(w.nds), id=w.id, highway = w.tags['highway'])


    for n_id in G.nodes_iter():
        n = osm.nodes[n_id]
        G.node[n_id] = dict(lon=n.lon,lat=n.lat)
    return G, osm

def plot_bokeh(osm):
    renderingRules = {
        'primary': dict(
                line_dash       = 'solid',
                line_width      = 6,
                line_cap        = 'round',
                color           ='#ee82ee',
                zorder          = -1,
                ),
        'primary_link': dict(
                line_dash       = 'solid',
                line_width       = 6,
                line_cap        = 'round',
                color           = '#da70d6',
                zorder          = -1,
                ),
        'secondary': dict(
                line_dash       = 'solid',
                line_width       = 6,
                line_cap        = 'round',
                color           = '#d8bfd8',
                zorder          = -2,
                ),
        'secondary_link': dict(
                line_dash       = 'solid',
                line_width       = 6,
                line_cap        = 'round',
                color           = '#d8bfd8',
                zorder          = -2,
                ),
        'tertiary': dict(
                line_dash       = 'solid',
                line_width       = 4,
                line_cap        = 'round',
                color           = (0.0,0.0,0.7),
                zorder          = -3,
                ),
        'tertiary_link': dict(
                line_dash       = 'solid',
                line_width       = 4,
                line_cap        = 'round',
                color           = (0.0,0.0,0.7),
                zorder          = -3,
                ),
        'residential': dict(
                line_dash       = 'solid',
                line_width       = 1,
                line_cap        = 'round',
                color           = (0.1,0.1,0.1),
                zorder          = -99,
                ),
        'unclassified': dict(
                line_dash       = 'solid',
                line_width       = 1,
                line_cap        = 'round',
                color           = (0.5,0.5,0.5),
                zorder          = -1,
                ),
        'default': dict(
                line_dash       = 'solid',
                line_width       = 3,
                line_cap        = 'round',
                color           = 'b',
                zorder          = -1,
                ),
        }

    # get bounds from OSM data
    minX = float(osm.bounds['minlon'])
    maxX = float(osm.bounds['maxlon'])
    minY = float(osm.bounds['minlat'])
    maxY = float(osm.bounds['maxlat'])

    # create a new plot
    p = plt.figure(
       tools="pan,box_zoom,reset,save",
       y_axis_type="log", title="OSM map",
    )

    plt.output_file("osm.html")

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

            if wayType in list(renderingRules.keys()):
                thisRendering = renderingRules[wayType]
            else:
                thisRendering = renderingRules['default']

            for nCnt, nID in enumerate(osm.ways[nodeID].nds):
                y = float(osm.nodes[nID].lat)
                x = float(osm.nodes[nID].lon)

                if oldX is None:
                    pass
                else:
                    p.line([oldX, x], [oldY, y],
                           line_dash  = thisRendering['line_dash'],
                           line_width = thisRendering['line_width'],
                           color      = thisRendering['color'],
                           line_cap   = thisRendering['line_cap'])

                oldX = x
                oldY = y

    print("Done bokeh")
    plt.show(p)

def matplot(osm):
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
        'default': dict(
                linestyle       = '-',
                linewidth       = 3,
                color           = 'b',
                zorder          = -1,
                ),
        }

    # get bounds from OSM data
    minX = float(osm.bounds['minlon'])
    maxX = float(osm.bounds['maxlon'])
    minY = float(osm.bounds['minlat'])
    maxY = float(osm.bounds['maxlat'])

    import pylab as plt
    fig = plt.figure()
    ax = fig.add_subplot(111,autoscale_on=False,xlim=(minX,maxX),ylim=(minY,maxY))

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

            if wayType in list(renderingRules.keys()):
                thisRendering = renderingRules[wayType]
            else:
                thisRendering = renderingRules['default']

            for nCnt, nID in enumerate(osm.ways[nodeID].nds):
                y = float(osm.nodes[nID].lat)
                x = float(osm.nodes[nID].lon)

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
                            )

                oldX = x
                oldY = y

    plt.show()

def main():
    graph, osm = read_osm(sys.argv[1])
    print(osm.bounds)
    plot_bokeh(osm)
    matplot(osm)

if __name__ == "__main__":
    main()