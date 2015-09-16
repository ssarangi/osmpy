#############################################################
## from http://code.activestate.com/recipes/534109-xml-to-python-data-structure/
 
import re
import xml.sax.handler
from bokeh.plotting import figure, output_file, show
 
def xml2obj(src):
    """
    A simple function to converts XML data into native Python object.
    """
 
    non_id_char = re.compile('[^_0-9a-zA-Z]')
    def _name_mangle(name):
        return non_id_char.sub('_', name)
 
    class DataNode(object):
        def __init__(self):
            self._attrs = {}    # XML attributes and child elements
            self.data = None    # child text data

        def __len__(self):
            # treat single element as a list of 1
            return 1

        def __getitem__(self, key):
            if isinstance(key, str):
                return self._attrs.get(key,None)
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
            return self._attrs.get(name,None)

        def _add_xml_attr(self, name, value):
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
            return '{%s}' % ', '.join(['%s:%s' % (k,repr(v)) for k,v in items])
 
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
                self.current._add_xml_attr(_name_mangle(k), v)

        def endElement(self, name):
            text = ''.join(self.text_parts).strip()
            if text:
                self.current.data = text
            if self.current._attrs:
                obj = self.current
            else:
                # a text only node is simply represented by the string
                obj = text or ''
            self.current, self.text_parts = self.stack.pop()
            self.current._add_xml_attr(_name_mangle(name), obj)

        def characters(self, content):
            self.text_parts.append(content)
 
    builder = TreeBuilder()
    if isinstance(src,str):
        xml.sax.parseString(src, builder)
    else:
        xml.sax.parse(src, builder)
    return list(builder.root._attrs.values())[0]
 
 
 
#############################################################
 
 
 
def render(myMap):
    # make dictionary of node IDs
    nodes = {}
    for node in myMap['node']:
        nodes[node['id']] = node
         
    ways = {}
    for way in myMap['way']:
        ways[way['id']]=way

    import pylab as p

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
    minX = float(myMap['bounds']['minlon'])
    maxX = float(myMap['bounds']['maxlon'])
    minY = float(myMap['bounds']['minlat'])
    maxY = float(myMap['bounds']['maxlat'])
 
 
 
    fig = p.figure()
     
    # by setting limits before hand, plotting is about 3 times faster
    ax = fig.add_subplot(111,autoscale_on=False,xlim=(minX,maxX),ylim=(minY,maxY))
     
    for idx,nodeID in enumerate(ways.keys()):
        wayTags         = ways[nodeID]['tag']
        if not wayTags==None:
            hwyTypeList  = [d['v'] for d in wayTags if d['k']=='highway']
            if len(hwyTypeList)>0:
                    wayType = hwyTypeList[0]  
            else:
                    wayType = None
        else:
            wayType = None
        try:
            if wayType in ['primary','primary_link',
                            'unclassified',
                            'secondary','secondary_link',
                            'tertiary','tertiary_link',
                            'residential',
                            'trunk','trunk_link',
                            'motorway','motorway_link',
                            ]:
                oldX = None
                oldY = None
                 
                if wayType in list(renderingRules.keys()):
                    thisRendering = renderingRules[wayType]
                else:
                    thisRendering = renderingRules['default']
                     
                for nCnt,nID in enumerate(ways[nodeID]['nd']):
                    y = float(nodes[nID['ref']]['lat'])
                    x = float(nodes[nID['ref']]['lon'])
                    if oldX == None:
                        pass
                    else:
                        p.plot([oldX,x],[oldY,y],
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
                        
        except KeyError:
            pass
           
    print('Done Plotting')
    p.show()
 
 
########################################

import sys
src = open(sys.argv[1])
myMap = xml2obj(src)
render(myMap)