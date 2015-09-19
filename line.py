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

W, H = 400, 400


i_scale_factor = 10000

VERT_SHADER = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
attribute vec3 a_position;
void main (void) {
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""

FRAG_SHADER = """
varying float v_id;
void main()
{
   gl_FragColor = vec4(1,0,0,1);
}
"""


class Canvas(app.Canvas):

    # ---------------------------------
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(W, H))

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.l_bb_center = []

        vbuffer = np.array([
             [ 0.6459411621,  0.4028605652,   0.1       ],
             [ 0.6459389496,  0.4028595352,   0.1       ],
             [ 0.6459381104,  0.4028591156,   0.1       ],
             [ 0.6459369659,  0.4028584671,   0.1       ],
             [ 0.6459358978,  0.4028577423,   0.1       ],
             [ 0.6459277344,  0.4028533173,   0.1       ],
             [ 0.6459257507,  0.402852211 ,   0.1       ],
             [ 0.6459215546,  0.402849884 ,   0.1       ],
             [ 0.6459153748,  0.402846489 ,   0.1       ],
             [ 0.6459139252,  0.4028466415,   0.1       ],
             [ 0.6459128571,  0.4028461456,   0.1       ],
             [ 0.6459088135,  0.4028437805,   0.1       ],
             [ 0.6459063721,  0.4028423691,   0.1       ],
             [ 0.6459036255,  0.4028408813,   0.1       ],
             [ 0.6459024811,  0.4028401947,   0.1       ],
             [ 0.6459013367,  0.4028395844,   0.1       ],
             [ 0.6458982086,  0.402837944 ,   0.1       ],
             [ 0.6458907318,  0.4028339005,   0.1       ],
             [ 0.6458789825,  0.4028277206,   0.1       ],
             [ 0.645867157 ,  0.4028213882,   0.1       ],
             [ 0.6458557892,  0.402815094 ,   0.1       ],
             [ 0.6458435822,  0.4028086472,   0.1       ]]).astype(np.float32)

        arr_min = np.full(vbuffer.shape, vbuffer.min(axis=0))
        arr_max = np.full(vbuffer.shape, vbuffer.max(axis=0))

        arr_bounded = vbuffer - arr_min

        arr_bounded = arr_bounded * vbuffer.max(axis=0) * i_scale_factor

        arr_bounded = arr_bounded.astype(np.float32)
        print arr_bounded

        self.l_bb_center = (vbuffer.max(axis=0) - vbuffer.min(axis=0)) * i_scale_factor / 2.0

        print self.l_bb_center

        # Set uniform and attribute
        self.program['a_position'] = gloo.VertexBuffer(arr_bounded)

        self.translate = 5
        self.view = translate((-self.l_bb_center[0], -self.l_bb_center[1], -self.translate), dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)

        # self.projection = ortho(-1, 1, -1, 1, -10, 10)
        self.program['u_projection'] = self.projection

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view

        self.theta = 0
        self.phi = 0

        self.context.set_clear_color('white')
        self.context.set_state('translucent')

        self.timer = app.Timer('auto', connect=self.on_timer)

        self.show()

    # ---------------------------------
    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

    # ---------------------------------
    def on_timer(self, event):
        self.theta += .5
        self.phi += .5
        self.model = np.dot(rotate(self.theta, (0, 0, 1)),
                            rotate(self.phi, (0, 1, 0)))
        self.program['u_model'] = self.model
        self.update()

    # ---------------------------------
    def on_resize(self, event):
        gloo.set_viewport(0, 0, event.physical_size[0], event.physical_size[1])
        self.projection = perspective(45.0, event.size[0] /
                                      float(event.size[1]), 1.0, 1000.0)

        # self.projection = ortho(-1, 1, -1, 1, -10, 10)
        self.program['u_projection'] = self.projection

    # ---------------------------------
    def on_mouse_wheel(self, event):
        self.translate += event.delta[1]
        self.translate = max(2, self.translate)
        self.view = translate((-self.l_bb_center[0], -self.l_bb_center[1], -self.translate))
        self.program['u_view'] = self.view
        self.update()

    # ---------------------------------
    def on_draw(self, event):
        self.context.clear()
        self.program.draw('line_strip')


if __name__ == '__main__':
    c = Canvas()
    app.run()