#############################################################
## from http://code.activestate.com/recipes/534109-xml-to-python-data-structure/

import re
import xml.sax.handler
import networkx as nx

def xml2obj(src):
    """
    A simple function to converts XML data into native Python object.
    """

    non_id_char = re.compile('[^_0-9a-zA-Z]')

    def _name_mangle(name):
        return non_id_char.sub('_', name)

    class DataNode(object):
        def __init__(self):
            self._attrs = {}  # XML attributes and child elements
            self.data = None  # child text data

        @property
        def attrs(self):
            return self._attrs

        def __len__(self):
            # treat single element as a list of 1
            return 1

        def __getitem__(self, key):
            if isinstance(key, str):
                return self._attrs.get(key, None)
            else:
                return [self][key]

        def __contains__(self, name):
            return name in self._attrs

        def __bool__(self):
            return bool(self._attrs or self.data)

        def __getattr__(self, name):
            if name.startswith('__'):
                # need to do this for Python special methods???
                raise AttributeError(name)
            return self._attrs.get(name, None)

        def add_xml_attr(self, name, value):
            if name in self._attrs:
                # multiple attribute of the same name are represented by a list
                children = self._attrs[name]
                if not isinstance(children, list):
                    children = [children]
                    self._attrs[name] = children
                children.append(value)
            else:
                self._attrs[name] = value

        def __str__(self):
            return self.data or ''

        def __repr__(self):
            items = sorted(self._attrs.items())
            if self.data:
                items.append(('data', self.data))
            return '{%s}' % ', '.join(['%s:%s' % (k, repr(v)) for k, v in items])

    class TreeBuilder(xml.sax.handler.ContentHandler):
        def __init__(self):
            self.stack = []
            self.root = DataNode()
            self.current = self.root
            self.text_parts = []

        def startElement(self, name, attrs):
            self.stack.append((self.current, self.text_parts))
            self.current = DataNode()
            self.text_parts = []
            # xml attributes --> python attributes
            for k, v in list(attrs.items()):
                self.current.add_xml_attr(_name_mangle(k), v)

        def endElement(self, name):
            text = ''.join(self.text_parts).strip()
            if text:
                self.current.data = text
            if self.current.attrs:
                obj = self.current
            else:
                # a text only node is simply represented by the string
                obj = text or ''
            self.current, self.text_parts = self.stack.pop()
            self.current.add_xml_attr(_name_mangle(name), obj)

        def characters(self, content):
            self.text_parts.append(content)

    builder = TreeBuilder()
    if isinstance(src, str):
        xml.sax.parseString(src, builder)
    else:
        xml.sax.parse(src, builder)
    return list(builder.root.attrs.values())[0]

#######################################################################################################################
'''
Now we need to convert it to a networkx node structure so we can verify a lot of things.
{'uid': '201724', 'changeset': '7047098', 'id': '65512120', 'version': '6', 'lat': '37.3327707', 'user': 'mk408',
'visible': 'true', 'timestamp': '2011-01-22T05:29:54Z', 'lon': '-121.8924944'}

Node structure
'''
class OSMNode:
    def __init__(self, id, lat, lon, visible):
        self._id = id
        self._lat = lat
        self._lon = lon
        self._visible = visible

    @property
    def id(self):
        return self._id

    @property
    def lat(self):
        return self._lat

    @property
    def lon(self):
        return self._lon

    @property
    def visible(self):
        return self._visible

    def __str__(self):
        return self._id

    __repr__ = __str__

'''
Way description extracted from OSM xml
{'tag': [{k:'created_by', v:'Potlatch 0.10'}, {k:'highway', v:'footway'}], 'id': '24168454', 'uid': '28775',
'changeset': '1757744', 'visible': 'true', 'nd': [{ref:'435801859'}, {ref:'261770813'}, {ref:'261771044'}],
'user': 'StellanL', 'timestamp': '2009-07-07T00:03:58Z', 'version': '5'}
'''

class OSMWay:
    useful_tags = ['highway', 'name', 'lanes', 'oneway', 'maxspeed']

    def __init__(self, id, nd_list, tags):
        self._id = id
        self._nd_list = nd_list

        for tag in tags:
            if tag.attrs['k'] in OSMWay.useful_tags:
                setattr(self, tag.attrs['k'], tag.attrs['v'])

    @property
    def id(self):
        return self._id

    @property
    def nodes(self):
        return self._nd_list

    def __str__(self):
        return self._id

def render(osm_map):
    # make dictionary of node IDs
    nodes = {}
    osmnodes = {}
    for node in osm_map['node']:
        osmnodes[node['id']] = OSMNode(node['id'], node['lat'], node['lon'], node['visible'])
        nodes[node['id']] = node

    ways = {}
    osmways = {}
    for way in osm_map['way']:
        w = OSMWay(way['id'], way['nd'], way['tag'])
        osmways[way['id']] = w
        ways[way['id']] = way

    graph = nx.Graph()

    for way in osmways.items():
        num_nodes = len(way.nodes)

        for i in range(0, num_nodes - 2):
            graph.add_node(osmways.items())

    import pylab as p

    rendering_rules = {
        'primary': dict(
            linestyle='-',
            linewidth=6,
            color='#ee82ee',
            zorder=-1,
        ),
        'primary_link': dict(
            linestyle='-',
            linewidth=6,
            color='#da70d6',
            zorder=-1,
        ),
        'secondary': dict(
            linestyle='-',
            linewidth=6,
            color='#d8bfd8',
            zorder=-2,
        ),
        'secondary_link': dict(
            linestyle='-',
            linewidth=6,
            color='#d8bfd8',
            zorder=-2,
        ),
        'tertiary': dict(
            linestyle='-',
            linewidth=4,
            color=(0.0, 0.0, 0.7),
            zorder=-3,
        ),
        'tertiary_link': dict(
            linestyle='-',
            linewidth=4,
            color=(0.0, 0.0, 0.7),
            zorder=-3,
        ),
        'residential': dict(
            linestyle='-',
            linewidth=1,
            color=(0.1, 0.1, 0.1),
            zorder=-99,
        ),
        'unclassified': dict(
            linestyle=':',
            linewidth=1,
            color=(0.5, 0.5, 0.5),
            zorder=-1,
        ),
        'default': dict(
            linestyle='-',
            linewidth=3,
            color='b',
            zorder=-1,
        ),
    }

    # get bounds from OSM data
    min_x = float(osm_map['bounds']['minlon'])
    max_x = float(osm_map['bounds']['maxlon'])
    min_y = float(osm_map['bounds']['minlat'])
    max_y = float(osm_map['bounds']['maxlat'])

    fig = p.figure()
    # by setting limits before hand, plotting is about 3 times faster
    fig.add_subplot(111, autoscale_on=False, xlim=(min_x, max_x), ylim=(min_y, max_y))

    for idx, nodeID in enumerate(ways.keys()):
        way_tags = ways[nodeID]['tag']
        if way_tags is not None:
            hwy_type_list = [d['v'] for d in way_tags if d['k'] == 'highway']
            if len(hwy_type_list) > 0:
                way_type = hwy_type_list[0]
            else:
                way_type = None
        else:
            way_type = None
        try:
            if way_type in ['primary', 'primary_link',
                            'unclassified',
                            'secondary', 'secondary_link',
                            'tertiary', 'tertiary_link',
                            'residential',
                            'trunk', 'trunk_link',
                            'motorway', 'motorway_link',
                            ]:
                old_x = None
                old_y = None

                if way_type in list(rendering_rules.keys()):
                    this_rendering = rendering_rules[way_type]
                else:
                    this_rendering = rendering_rules['default']

                for nCnt, nID in enumerate(ways[nodeID]['nd']):
                    y = float(nodes[nID['ref']]['lat'])
                    x = float(nodes[nID['ref']]['lon'])
                    if old_x is None:
                        pass
                    else:
                        p.plot([old_x, x], [old_y, y],
                               marker='',
                               linestyle=this_rendering['linestyle'],
                               linewidth=this_rendering['linewidth'],
                               color=this_rendering['color'],
                               solid_capstyle='round',
                               solid_joinstyle='round',
                               zorder=this_rendering['zorder'],
                               )
                    old_x = x
                    old_y = y

        except KeyError:
            pass

    print('Done Plotting')
    # p.axis('off')
    p.show()


import sys

src = open(sys.argv[1])
myMap = xml2obj(src)
render(myMap)
