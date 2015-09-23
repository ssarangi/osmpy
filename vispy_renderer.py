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
attribute vec2 a_texcoord;

// Varyings
varying vec2 v_texcoord;

void main (void) {
    v_texcoord = a_texcoord;
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
"""

FRAG_SHADER = """
varying float v_id;
uniform vec3 color;
uniform sampler2D u_texture;
varying vec2 v_texcoord;

void main()
{
    gl_FragColor = texture2D(u_texture, v_texcoord);
    gl_FragColor.a = 1.0;
    // gl_FragColor = vec4(color, 1.0) * gl_FragColor;
    gl_FragColor = mix(gl_FragColor, vec4(color, 1.0), 0.45);
}
"""

class Canvas(app.Canvas):

    # ---------------------------------
    def __init__(self, vbos, bbox, scale):
        app.Canvas.__init__(self, keys='interactive', fullscreen=False, size=(800.0, 800.0), vsync=True)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        img = misc.imread('bullseye.png')
        self.bullseye = gloo.Texture2D(img)
        self.bullseye.interpolation = 'linear'
        self.bullseye.wrapping = 'repeat'
        self.wireframe = False
        self.zoom = 0

        self.vbos = []
        self.tex_coords = []
        for vbo_info in vbos:
            vbo = vbo_info[0]
            color = vbo_info[1]

            self.l_bb_center = []

            vbuffer = np.array(vbo)
            arr_min = np.full(vbuffer.shape, [bbox[0], bbox[1], 0.0])
            arr_bounded = vbuffer - arr_min

            arr_bounded *= scale

            arr_bounded = arr_bounded.astype(np.float32)

            self.l_bb_center = (np.array([bbox[2], bbox[3]]) - np.array([bbox[0], bbox[1]])) * scale / 2.0

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
                pt_offset = self.extrude_point(point[0], point[1], curr_ax, 0.02)
                new_vbo.append(point)
                new_vbo.append(pt_offset)

            #Create IBO
            ibo_template = np.array([0,1,2,2,1,3])
            full_ibo = np.array([ibo_template + (i * 4) for i in range(0, (len(vbo)) - 1)])
            full_ibo = full_ibo.flatten().astype(np.uint32)
            # Set uniform and attribute

            tex_coords_fractions = np.array(range(0, int(len(new_vbo)/2)))
            tex_coords_fractions = tex_coords_fractions / (len(new_vbo)/2 - 1.0)

            lower_row = np.full(tex_coords_fractions.shape, [0])
            upper_row = np.full(tex_coords_fractions.shape, [1])

            lower_tuple = list(zip(lower_row, tex_coords_fractions))
            upper_tuple = list(zip(upper_row, tex_coords_fractions))

            uv_coords = list(zip(lower_tuple, upper_tuple))


            # tex_coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.float32)
            self.vbos.append((new_vbo, uv_coords, color, full_ibo))

            self.translate = 5.0
            self.view = translate((-self.l_bb_center[0], -self.l_bb_center[1], -self.translate), dtype=np.float32)
            self.model = np.eye(4, dtype=np.float32)

            gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
            self.projection = perspective(45.0, self.size[0] /
                                          float(self.size[1]), 1.0, 1000.0)

            self.program['u_projection'] = self.projection
            self.program['u_model'] = self.model
            self.program['u_view'] = self.view
            self.program['u_texture'] = self.bullseye

            self.theta = 0
            self.phi = 0

            self.context.set_clear_color('white')
            self.context.set_state('translucent')

            # self.timer = app.Timer('auto', connect=self.on_timer)
            self._timer = app.Timer('auto', connect=self.update, start=True)

            self.show()

    def extrude_point(self, curX, curY, axisVector, scale):
        extrudedPoint = [curX, curY, scale, 1]
        rotatedExtrudedPoint = [0, 0, 0]
        axisVector = self.normalizeVec(axisVector)
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

    def normalizeVec(self, axis):
        import math
        length = math.sqrt((axis[0] * axis[0]) + (axis[1] * axis[1]))
        return [axis[0]/length, axis[1]/length]


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
        if self.zoom < -4:
            self.zoom = -4
            # return
        elif self.zoom > 10:
            self.zoom = 10
            return

        self.translate += event.delta[1]
        self.translate = max(-1, self.translate)
        self.view = translate((-self.l_bb_center[0], -self.l_bb_center[1], -self.translate))
        self.program['u_view'] = self.view
        self.update()

    # ---------------------------------
    def on_draw(self, event):
        self.context.clear()
        if self.wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        for vbo_info in self.vbos:
            vbo = np.array(vbo_info[0]).astype(np.float32)
            tex_coords = vbo_info[1]
            color = vbo_info[2]            # Set uniform and attribute
            ibo = vbo_info[3]
            index_buffer = gloo.IndexBuffer(ibo)
            self.program['a_position'] = gloo.VertexBuffer(vbo)
            self.program['a_texcoord'] = tex_coords
            self.program['color'] = color
            self.program.draw('triangles', index_buffer)


if __name__ == '__main__':
    c = Canvas()
    app.run()