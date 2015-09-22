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

i_scale_factor = 100
# i_scale_factor = 1

VERT_SHADER = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
attribute vec3 a_position;
void main (void) {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
"""

FRAG_SHADER = """
varying float v_id;
uniform vec3 color;
void main()
{
   gl_FragColor = vec4(color, 1);
}
"""

class Canvas(app.Canvas):

    # ---------------------------------
    def __init__(self, vbos, bbox):
        app.Canvas.__init__(self, keys='interactive', fullscreen=False, size=(800.0, 800.0))
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        self.vbos = []
        for vbo_info in vbos:
            vbo = vbo_info[0]
            color = vbo_info[1]

            self.l_bb_center = []

            vbuffer = np.array(vbo)
            arr_min = np.full(vbuffer.shape, [bbox[0], bbox[1], 0.0])
            arr_bounded = vbuffer - arr_min

            arr_bounded *= i_scale_factor

            arr_bounded = arr_bounded.astype(np.float32)

            self.l_bb_center = (np.array([bbox[2], bbox[3]]) - np.array([bbox[0], bbox[1]])) * i_scale_factor / 2.0

            new_vbo = []
            axis = zip(arr_bounded, arr_bounded[1:])
            ax = [[ax[1][0] - ax[0][0], ax[1][1] - ax[0][1]] for ax in axis]
            old_ax = []
            new_arr_bounded = []

            for i in range(0, len(arr_bounded)):
                if i==0 or i==len(arr_bounded)-1:
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
                pt_offset = self.extrude_point(point[0], point[1], curr_ax, 0.03)
                new_vbo.append(point)
                new_vbo.append(pt_offset)
                old_ax = curr_ax
            # new_vbo.append(arr_bounded[-1])
            # last_point = arr_bounded[-1]
            # new_vbo.append(self.extrude_point(last_point[0], last_point[1], old_ax, 0.1))

            # Set uniform and attribute
            self.vbos.append((new_vbo, color))

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

            self.timer = app.Timer('auto', connect=self.on_timer)

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
        self.translate += event.delta[1]
        self.translate = max(-1, self.translate)
        self.view = translate((-self.l_bb_center[0], -self.l_bb_center[1], -self.translate))
        self.program['u_view'] = self.view
        self.update()

    # ---------------------------------
    def on_draw(self, event):
        self.context.clear()
        for vbo_info in self.vbos:
            vbo = vbo_info[0]
            color = vbo_info[1]            # Set uniform and attribute
            self.program['a_position'] = gloo.VertexBuffer(vbo)
            self.program['color'] = color
            self.program.draw('triangle_strip')


if __name__ == '__main__':
    c = Canvas()
    app.run()