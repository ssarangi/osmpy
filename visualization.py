import overpass
import urllib
from bs4 import BeautifulSoup
import IPython
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
import sys

sys.setrecursionlimit(500)

sys.path.append('./utm/')

from utm.conversion import from_latlon
import requests
api = overpass.API()


from vispy import gloo, app, visuals, scene
from vispy.util.transforms import perspective, translate, rotate
from vispy.visuals import transforms

from itertools import cycle

n = 500
a_position = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
a_id = np.random.randint(0, 30, (n, 1))
a_id = np.sort(a_id, axis=0).astype(np.float32)

class Canvas(scene.SceneCanvas):

    # ---------------------------------
    def __init__(self, buildings):
        app.Canvas.__init__(self, keys='interactive')
        self.size = (800, 800)

        self.visuals = []

        for b in buildings:
            v = visuals.PolygonVisual(pos=b.utm_scaled, color=(0.8, .2, 0, 1),
                                                    border_color=(1, 1, 1, 1))
            v.transform = transforms.STTransform(scale=(1, 1), translate=(0, 0))
            v2 = visuals.MarkersVisual()
            v2.set_data(pos=b.all_points, symbol="+", size=2)
            v2.transform = transforms.STTransform(scale=(1, 1), translate=(0, 0))

            self.visuals.append(v)
            self.visuals.append(v2)

        for v in self.visuals:
            v.tr_sys = transforms.TransformSystem(self)
            v.tr_sys.visual_to_document = v.transform

    # ---------------------------------
    def on_resize(self, event):

        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.projection = perspective(45.0, width / float(height), 1.0, 1000.0)

    # ---------------------------------
    def on_mouse_wheel(self, event):
        print(transforms.TransformSystem(self))
        return
        self.translate += event.delta[1]
        self.translate = max(2, self.translate)
        self.view = np.eye(4, dtype=np.float32)
        translate(self.view, 0, 0, -self.translate)
        self.program['u_view'] = self.view
        self.update()

    # ---------------------------------
    def on_draw(self, event):
        gloo.set_clear_color((1, 1, 0, 1))
        gloo.set_viewport(0, 0, *self.size)
        gloo.clear()
        for vis in self.visuals:
            vis.draw(vis.tr_sys)


QUERY_TMPL = "http://overpass-api.de/api/map?bbox={0},{1},{2},{3}"
query = QUERY_TMPL.format(6.08051,50.77248,6.08545,50.77662)

response = ""
try:
    with open("map.xml", mode="r", encoding="utf8") as f:
        response = f.read()
        f.close()
except:
    print("Nothing cached, ", query)
    xml_response = requests.get(query)
    response = xml_response.text
    f = open(query.replace('/', '')  + ".xml", "w+")
    f.write(response)

soup = BeautifulSoup(response)

nodes = soup.find_all("node")
buildings_raw = [el.parent for el in soup.select("tag[k=\"building\"]")]

class Building:
    def get_nodes(self, root, soup_tag):
        nodes = soup_tag.find_all('nd')
        self.coords_utm = []
        realnodes = [root.select("node[id=\"{0}\"]".format(n["ref"]))[0] for n in nodes]
        for node in realnodes:
            ll_tuple = (node["lat"], node["lon"])
            ll_utm = from_latlon(float(node["lat"]), float(node["lon"]))
            self.coords_utm.append((ll_utm[0], ll_utm[1]))
            self.coords_lat_lon.append(ll_tuple)
        self.coords_arr = np.array(self.coords_lat_lon)
        self.utm_arr = np.array(self.coords_utm)


    def __init__(self, xml_tag, root):
        self.coords_lat_lon = []
        self.get_nodes(root, xml_tag)

    def generate_points(self):
        points = []
        self.all_points = None
        it = cycle(self.utm_scaled)
        shape = self.utm_scaled.shape
        arr_len = shape[0]
        max_point_dist = 5
        i = 0
        for i in range(-1, arr_len -1):
            c = self.utm_scaled[i]
            c_next = self.utm_scaled[i + 1]
            l = np.linalg.norm(c_next - c)
            ev = (c_next - c) / l
            n_step = np.floor(l / max_point_dist)
            x = np.linspace(0, l, n_step)

            points_x = (ev[0] * x)[np.newaxis, :].T
            points_y = (ev[1] * x)[np.newaxis, :].T

            points = np.hstack((points_x, points_y))

            points = points + c
            if self.all_points is not None:
                self.all_points = np.vstack((self.all_points, points))
            else:
                self.all_points = points
            if i == arr_len - 1:
                break
            i += 1
        # print(self.all_points)


Buildings = []
try:
    # raise NotImplementedError()
    with open('buildings_cache' + query.replace('/', '') + '.picke', 'rb') as pc:
        Buildings = pickle.load(pc)
except:
    for b in buildings_raw:
        print("creating buidling")
        B = Building(b, soup)
        Buildings.append(B)

    with open('buildings_cache' + query.replace('/', '') + '.picke', 'wb+') as pc:
        pickle.dump(Buildings, pc)

for b in Buildings:
    if b.utm_arr.shape[0] == 0:
        Buildings.remove(b)
all_buildings = None
all_id = None
for b in Buildings:
    if b.utm_arr == None:
        continue
    # plt.plot(b.utm_arr[:,0], b.utm_arr[:,1])
    if all_buildings is not None and b.utm_arr is not None:
        print(all_buildings.shape, b.utm_arr.shape)
        all_buildings = np.vstack((all_buildings, b.utm_arr))
        ids = np.ones(b.utm_arr.shape[0])
        ids[-1] = 0
        all_id = np.hstack((all_id, ids))
    else:
        # print("SHape", all_buildings.shape)

        all_buildings = b.utm_arr
        ids = np.ones(b.utm_arr.shape[0])
        ids[-1] = 1
        all_id = ids

# from http://stackoverflow.com/questions/12443688/calculating-bounding-box-of-numpy-array
def bounding_box(iterable):
    min_x, min_y = np.min(iterable, axis=0)
    max_x, max_y = np.max(iterable, axis=0)
    return np.array([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

bb = bounding_box(all_buildings)
all_buildings = all_buildings - bb[0]
for b in Buildings:
    b.utm_scaled = (b.utm_arr - bb[0])

bb = bounding_box(all_buildings)

all_buildings_scaled = np.divide(all_buildings, bb[2])
all_buildings_scaled = np.append(all_buildings_scaled, np.zeros((all_buildings.shape[0], 1)), axis=1)
a_position = all_buildings_scaled.astype(np.float32)

for b in Buildings:
    b.utm_scaled = np.divide(b.utm_scaled, bb[2]) * 800
    b.generate_points()

n = all_buildings.shape[0]
# a_id = np.random.randint(0, 30, (n, 1))
# a_id = np.sort(a_id, axis=0).astype(np.float32)
a_id = all_id.astype(np.float32)


# import vispy.plot as vp

# fig = vp.Fig(show=False)


all_points = None
for b in Buildings:
    if all_points is not None:
        all_points = np.vstack((all_points, b.all_points))
    else:
        all_points = b.all_points

# fig[1,1].plot(all_points)
# fig.show()

corners = []

for b in Buildings:
    corners += b.utm_scaled.tolist()

corny_arr = np.array(corners)

plt.scatter(corny_arr[:, 0], corny_arr[:, 1])
plt.show()
np.savetxt('corners.csv', corny_arr)
c = Canvas(Buildings)
c.show()
app.run()