# !/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 2
""" Show a bunch of lines.
This example demonstrates how multiple line-pieces can be drawn
using one call, by discarting some fragments.

Note that this example uses canvas.context.X() to call gloo functions.
These functions are also available as vispy.gloo.X(), but apply
explicitly to the canvas. We still need to decide which we think is the
preferred API.
"""

import numpy as np
from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective, translate, rotate, ortho
import OpenGL.GL as gl
from scipy import misc

VERT_SHADER = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
attribute vec3 a_position;
// attribute vec2 a_texcoord;
attribute float point_size;

// Varyings
// varying vec2 v_texcoord;

void main (void) {
    // v_texcoord = a_texcoord;
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    gl_PointSize = point_size;
}
"""

FRAG_SHADER = """
varying float v_id;
uniform vec3 color;
// uniform sampler2D u_texture;
uniform int use_textures;
// varying vec2 v_texcoord;

void main()
{
/*
    if (use_textures == 1)
    {
        gl_FragColor = texture2D(u_texture, v_texcoord);
        gl_FragColor.a = 1.0;
        gl_FragColor = mix(gl_FragColor, vec4(color, 1.0), 0.45);
    }
    else
*/
        gl_FragColor = vec4(color, 1.0);
}
"""

def normalize_vec(axis):
    import math
    length = math.sqrt((axis[0] * axis[0]) + (axis[1] * axis[1]))
    return [axis[0]/length, axis[1]/length]

def extrude_point(curX, curY, axisVector, scale):
    extrudedPoint = [curX, curY, scale, 1]
    rotatedExtrudedPoint = [0, 0, 0]
    axisVector = normalize_vec(axisVector)
    a = curX
    b = curY
    u = axisVector[0]
    v = axisVector[1]
    x = extrudedPoint[0]
    y = extrudedPoint[1]
    z = extrudedPoint[2]

    rotatedExtrudedPoint[0] = (a * v * v ) -  (u * ((b * v) - (u * x) - (v * y)) + (v * z))
    rotatedExtrudedPoint[1] = (b * u * u) - (v * ((a * u) - (u * x) - (v * y)) - (u * z))
    rotatedExtrudedPoint[2] = (a * v) - (b * u) - (v * x) + (u * y)
    return rotatedExtrudedPoint

def generate_tex_coords(vbo):
    # Set uniform and attribute
    tex_coords_fractions = np.array(range(0, int(len(vbo)/2)))
    tex_coords_fractions = tex_coords_fractions / (len(vbo)/2 - 1.0)

    lower_row = np.full(tex_coords_fractions.shape, [0])
    upper_row = np.full(tex_coords_fractions.shape, [1])

    lower_tuple = list(zip(lower_row, tex_coords_fractions))
    upper_tuple = list(zip(upper_row, tex_coords_fractions))

    uv_coords = list(zip(lower_tuple, upper_tuple))
    return uv_coords

def generate_ibo(vbo):
    # Create IBO
    ibo_template = np.array([0,1,2,2,1,3])
    ibo = np.array([ibo_template + (i * 4) for i in range(0, (len(vbo)) - 1)])
    ibo = ibo.flatten().astype(np.uint32)
    return ibo

def cap_lines(p1, p2):
    th = np.linspace(np.pi / 2.0, -np.pi / 2.0, 100)
    center = (p1 + p2) / 2.0
    radius = np.linalg.norm(center - p1)
    x = center[0] + radius * np.cos(th)
    y = center[1] + radius * np.sin(th)
    vec = p2 - p1

def scale_to_bb(l_points, bb, i_scale):
    vbuffer = np.array(l_points)
    arr_min = np.full(vbuffer.shape, [bb[0], bb[1], 0.0])
    arr_bounded = vbuffer - arr_min

    arr_bounded *= i_scale

    arr_bounded = arr_bounded.astype(np.float32)
    return arr_bounded


def generate_vbo(arr_bounded):
    new_vbo = []
    axis = zip(arr_bounded, arr_bounded[1:])
    ax = [[ax[1][0] - ax[0][0], ax[1][1] - ax[0][1]] for ax in axis]
    new_arr_bounded = []

    for i in range(0, len(arr_bounded)):
        if i == 0 or i == len(arr_bounded)-1:
            new_arr_bounded.append(arr_bounded[i])
        else:
            new_arr_bounded.append(arr_bounded[i])
            new_arr_bounded.append(arr_bounded[i])

    new_ax = []
    for i in range(0, len(ax)):
         new_ax.append(ax[i])
         new_ax.append(ax[i])

    arr_bounded = new_arr_bounded
    ax = new_ax
    for i in range(0, len(arr_bounded)):
        point = arr_bounded[i]
        curr_ax = ax[i]
        pt_offset_pos = extrude_point(point[0], point[1], curr_ax, 0.01)
        pt_offset_neg = extrude_point(point[0], point[1], curr_ax, -0.01)
        new_vbo.append(pt_offset_neg)
        new_vbo.append(pt_offset_pos)

    return new_vbo

class Canvas(app.Canvas):

    # ---------------------------------
    def __init__(self, road_vbos, other_vbos, bbox, scale):
        app.Canvas.__init__(self, keys='interactive', fullscreen=False, size=(800.0, 800.0), vsync=True)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        img = misc.imread('bullseye.png')
        self.bullseye = gloo.Texture2D(img)
        self.bullseye.interpolation = 'linear'
        self.bullseye.wrapping = 'repeat'
        self.wireframe = False
        self.zoom = 0

        self.road_vbos = []
        self.road_ibo = []
        self.other_vbos = []
        self.single_vbo = []
        self.single_ibo = []
        self.tex_coords = []
        self.l_bb_center = (np.array([bbox[2], bbox[3]]) - np.array([bbox[0], bbox[1]])) * scale / 2.0

        flattened_vbo = []

        for vbo_info in road_vbos:
            vbo = vbo_info[0]
            color = vbo_info[1]
            arr_bounded = scale_to_bb(vbo, bbox, scale)
            new_vbo = generate_vbo(arr_bounded)
            uv_coords = generate_tex_coords(new_vbo)

            for p in new_vbo:
                self.road_vbos.append(p)

            for p in vbo:
                flattened_vbo.append(p)

        self.road_ibo = generate_ibo(flattened_vbo)
        point_count = 0
        for vbo_info in other_vbos:
            vbo = vbo_info[0]

            color = vbo_info[1]
            vbo = scale_to_bb(vbo, bbox, scale)
            self.other_vbos.append(gloo.VertexBuffer(vbo))
            start_point = point_count
            # first point
            self.single_ibo.append(point_count)

            for i,point in enumerate(vbo):
                self.single_vbo.append(point)
                if i > 0:
                    self.single_ibo.append(point_count)
                    self.single_ibo.append(point_count)
                point_count += 1

            self.single_ibo.append(start_point)


        self.single_vbo = gloo.VertexBuffer(self.single_vbo)
        self.single_ibo = gloo.IndexBuffer(self.single_ibo)

        self.road_vbo = gloo.VertexBuffer(self.road_vbos)
        self.road_ibo = gloo.IndexBuffer(self.road_ibo)

        self.translate = 5.0
        self.view = translate((-self.l_bb_center[0], -self.l_bb_center[1], -self.translate), dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)

        self.program['u_projection'] = self.projection
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view

        self.theta = 0
        self.phi = 0

        self.context.set_clear_color('white')
        self.context.set_state('translucent')

        self._timer = app.Timer('auto', connect=self.update, start=True)

        self.show()

    # ---------------------------------
    def on_key_press(self, event):
        if event.text == 'x':
            self.theta += .5
            self.phi += .5
            self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                                rotate(self.phi, (0, 1, 0)))
            self.program['u_model'] = self.model
            self.update()
        elif event.text == 'z':
            self.theta += .5
            self.phi += .5
            self.model = np.dot(rotate(-self.theta, (0, 0, 1)),
                                rotate(self.phi, (0, 1, 0)))
            self.program['u_model'] = self.model
            self.update()
        elif event.text == 'w':
            self.wireframe = not self.wireframe

        self.update()

    def on_mouse_press(self, event):
        pos_scene = event.pos[:3]
        print(pos_scene)
        bb_x = (self.l_bb_center[0] + pos_scene[0]) / abs(self.zoom)
        bb_y = (self.l_bb_center[1] + pos_scene[1]) / abs(self.zoom)

        print(bb_x, bb_y)

        # find closest point to mouse and select it
        # self.selected_point, self.selected_index = self.select_point(event)

        # if no point was clicked add a new one
        # if self.selected_point is None:
        #     print("adding point", len(self.pos))
        #     self._pos = np.append(self.pos, [pos_scene], axis=0)
        #     self.set_data(pos=self.pos)
        #     self.marker_colors = np.ones((len(self.pos), 4), dtype=np.float32)
        #     self.selected_point = self.pos[-1]
        #     self.selected_index = len(self.pos) - 1
        #
        # # update markers and highlights
        # self.update_markers(self.selected_index)


    # ---------------------------------
    def on_timer(self, event):
        pass

    # ---------------------------------
    def on_resize(self, event):
        gloo.set_viewport(0, 0, event.physical_size[0], event.physical_size[1])
        self.projection = perspective(45.0, event.size[0] /
                                      float(event.size[1]), 1.0, 100.0)

        self.program['u_projection'] = self.projection

    # ---------------------------------
    def on_mouse_wheel(self, event):
        self.zoom += event.delta[1]

        if self.zoom < 0 and event.delta[1] > 0 or self.zoom > 0 and event.delta[1] < 0:
            self.zoom = event.delta[1]

        if self.zoom < -4:
            self.zoom = -4
            # return
        elif self.zoom > 10:
            self.zoom = 10
            return

        self.translate += self.zoom / 10
        self.translate = max(-1, self.translate)
        self.view = translate((-self.l_bb_center[0], -self.l_bb_center[1], -self.translate))
        self.program['u_view'] = self.view
        self.update()

    # ---------------------------------
    def on_draw(self, event):
        self.context.clear(color=(0.3, 0.3, 0.3, 1.0))
        if self.wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        # Draw Buildings
        self.program['a_position'] = self.single_vbo
        # self.program['a_texcoord'] = tex_coords
        self.program['color'] = (1.0, 1.0, 0.0)
        self.program['point_size'] = 1
        # self.program['use_textures'] = 0
        self.program.draw('lines', self.single_ibo)


        # Draw Roads
        self.program['a_position'] = self.road_vbo
        #self.program['a_texcoord'] = tex_coords
        self.program['color'] = (0.0, 0.0, 0.7)
        self.program['point_size'] = 1
        # self.program['u_texture'] = self.bullseye
        # self.program['use_textures'] = 0
        self.program.draw('triangles', self.road_ibo)



if __name__ == '__main__':
    c = Canvas()
    app.run()