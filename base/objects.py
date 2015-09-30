"""
The MIT License (MIT)

Copyright (c) 2015 <Satyajit Sarangi, Pranabesh Sinha>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import xml.sax

class Node:
    def __init__(self, id, lon, lat):
        """
        Node class for OSM.
        :param id: id of the node
        :param lon: floating point
        :param lat: floating point
        :return: Node
        """
        self._id = id
        self._lon = lon
        self._lat = lat
        self._tags = {}

    @property
    def id(self):
        """
        :return: Return integer (id)
        """
        return self._id

    @property
    def lon(self):
        """
        :return: Return float (lon)
        """
        return self._lon

    @property
    def lat(self):
        """
        :return: Return float (lat)
        """
        return self._lat

    @property
    def tags(self):
        """
        :return: Return dictionary
        """
        return self._tags


class Way:
    def __init__(self, id):
        """
        Way class for OSM
        :param id: id of Way
        :return: Way
        """
        self._id = id
        self._nds = np.array()
        self._tags = {}

    @property
    def id(self):
        return self._id

    @property
    def nds(self):
        """
        :return: Returns a read only copy of the nodes for the Way
        """
        return self._nds

    @property
    def tags(self):
        return self._tags

    def add_node(self, nd):
        self._nds = np.append(self._nds, nd)


class Relation:
    def __init__(self, id):
        """
        Relation class for OSM
        :param id: id of Relation
        :return: Relation
        """
        self._id = id

    @property
    def id(self):
        return self._id

class BBox:
    """
    Bounding Box class. Holds the minimum longitude / latitude and the maximum longitude / latitude
    """
    def __init__(self, maxlon, minlon, maxlat, minlat):
        self._maxlon = maxlon
        self._minlon = minlon
        self._maxlat = maxlat
        self._minlat = minlat

    @property
    def maxlon(self):
        return self._maxlon

    @property
    def minlon(self):
        return self._minlon

    @property
    def maxlat(self):
        return self.maxlat

    @property
    def minlat(self):
        return self.minlat

class OSM:
    """
    Base class holding all the information about the nodes, ways and relations.
    """
    def __init__(self, filename_or_stream):
        nodes = {}
        ways = {}
        relations = {}
        bbox = [0]

        class XMLReader(xml.sax.ContentHandler):
            @classmethod
            def setDocumentLocator(self, locator):
                pass

            @classmethod
            def startDocument(self):
                pass

            @classmethod
            def endDocument(self):
                pass

            @classmethod
            def startElement(self, name, attrs):
                if name == 'node':
                    self.currElem = Node(attrs['id'], float(attrs['lon']), float(attrs['lat']))
                elif name == 'way':
                    self.currElem = Way(attrs['id'])
                elif name == 'tag':
                    self.currElem.tags[attrs['k']] = attrs['v']
                elif name == 'nd':
                    self.currElem.nds.append(attrs['ref'])
                elif name == 'bounds':
                    bbox[0] = BBox(float(attrs['maxlon']), float(attrs['minlon']), float(attrs['maxlat']), float(attrs['minlat']))
                elif name == 'relation':
                    self.currElem = Relation(attrs['id'])

            @classmethod
            def endElement(self, name):
                if name == 'node':
                    nodes[self.currElem.id] = self.currElem
                elif name == 'way':
                    ways[self.currElem.id] = self.currElem
                elif name == 'relation':
                    relations[self.currElem.id] = self.currElem

        xml.sax.parse(filename_or_stream, XMLReader)
        self._nodes = nodes
        self._ways = ways
        self._relations = relations
        self._bbox = bbox[0]

    @property
    def nodes(self):
        return self._nodes

    @property
    def ways(self):
        return self.ways

    @property
    def relations(self):
        return self.relations

    @property
    def bbox(self):
        return self._bbox