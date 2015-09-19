__author__ = 'sarangis'

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

# Create vertices
n = 100
a_position = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)


VERT_SHADER = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
attribute vec3 a_position;
// attribute float a_id;
// varying float v_id;
void main (void) {
    // v_id = a_id;
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
"""

FRAG_SHADER = """
// varying float v_id;
void main()
{
    // float f = fract(v_id);
    // The second useless test is needed on OSX 10.8 (fuck)
    /*
    if( (f > 0.0001) && (f < .9999) )
        discard;
    else */
        gl_FragColor = vec4(1,0,0,1);
}
"""


class Canvas(app.Canvas):

    # ---------------------------------
    def __init__(self, buffer):

       #  buffer = np.array([
       # [ -4.42757189e-01,  -5.16471803e-01,  -6.51616335e-01],
       # [  3.70873958e-01,   3.83630782e-01,   1.89464033e-01],
       # [ -9.23565388e-01,   8.84109557e-01,   8.74280035e-01],
       # [  7.06181884e-01,   1.11561500e-01,   4.63272840e-01],
       # [ -2.33703643e-01,   8.74389410e-01,  -7.08803475e-01],
       # [  9.14665759e-01,  -4.91853468e-02,   5.51214755e-01],
       # [ -3.51651721e-02,   9.53544497e-01,   1.13748061e-02],
       # [ -2.00554803e-01,   8.23601604e-01,  -2.01624975e-01],
       # [ -5.78361809e-01,  -7.19190955e-01,  -6.34798348e-01],
       # [  2.07641460e-02,  -9.39733744e-01,  -7.52322018e-01],
       # [ -3.88619483e-01,   6.74676895e-02,  -9.36758697e-01],
       # [  8.96090984e-01,   9.09111351e-02,  -3.39223385e-01],
       # [ -9.60957527e-01,   5.22359312e-01,  -9.29324806e-01]]).astype(np.float32)
    #
    #     buffer = np.array([
    #    [ -4.42757189e-01,  -5.16471803e-01, 0.0],
    #    [  3.70873958e-01,   3.83630782e-01, 0.0],
    #    [ -9.23565388e-01,   8.84109557e-01, 0.0],
    #    [  7.06181884e-01,   1.11561500e-01, 0.0],
    # [  8.96090984e-01,   9.09111351e-02, 0.0],
    #    [ -9.60957527e-01,   5.22359312e-01, 0.0]]).astype(np.float32)

       #  buffer = np.array([
       # [ -4.42757189e-01,  -5.164743803e-01, 0.0],
       # [  3.70873958e-01,   3.83630782e-01, 0.0]]
       #  ).astype(np.float32)
       #
       #  buffer = np.array([
       # [ 6.459411621e-01,  4.028605652e-01,   0.1       ],
       #  [ 6.453389496e-01,  4.028395352e-01,   0.1       ]]).astype(np.float32)

        print(buffer)
        # buffer = a_position
        a_id = np.random.randint(0, high=len(buffer) - 1, size=(len(buffer), 1))
        a_id = np.sort(a_id, axis=0).astype(np.float32)

        app.Canvas.__init__(self, keys='interactive', size=(W, H))

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)

        # Set uniform and attribute
        # self.program['a_id'] = gloo.VertexBuffer(a_id)
        self.program['a_position'] = gloo.VertexBuffer(buffer[0:3])

        self.translate = 5
        self.view = translate((0, 0, -self.translate), dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        # self.model = self.model * 10
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), -100.0, 1000.0)

        # self.projection = ortho(-200, 200, -200, 200, -100, 100)
        self.program['u_projection'] = self.projection

        buffer1 = np.array([[ 64.59411621,  40.28605652,   0.1, 0.0       ],
 [ 64.59389496,  40.28595352,   0.1, 0.0       ],
 [ 64.59381104,  40.28591156,   0.1, 0.0       ],
 [ 64.59369659,  40.28584671,   0.1, 0.0       ],
 [ 64.59358978,  40.28577423,   0.1, 0.0       ],
 [ 64.59277344,  40.28533173,   0.1, 0.0       ],
 [ 64.59257507,  40.2852211 ,   0.1, 0.0       ],
 [ 64.59215546,  40.2849884 ,   0.1, 0.0       ],
 [ 64.59153748,  40.2846489 ,   0.1, 0.0       ],
 [ 64.59139252,  40.28466415,   0.1, 0.0       ],
 [ 64.59128571,  40.28461456,   0.1, 0.0       ],
 [ 64.59088135,  40.28437805,   0.1, 0.0       ],
 [ 64.59063721,  40.28423691,   0.1, 0.0       ],
 [ 64.59036255,  40.28408813,   0.1, 0.0       ],
 [ 64.59024811,  40.28401947,   0.1, 0.0       ],
 [ 64.59013367,  40.28395844,   0.1, 0.0       ],
 [ 64.58982086,  40.2837944 ,   0.1, 0.0       ],
 [ 64.58907318,  40.28339005,   0.1, 0.0       ],
 [ 64.58789825,  40.28277206,   0.1, 0.0       ],
 [ 64.5867157 ,  40.28213882,   0.1, 0.0       ],
 [ 64.58557892,  40.2815094 ,   0.1, 0.0       ],
 [ 64.58435822,  40.28086472,   0.1, 0.0       ],
 [ 64.58195496,  40.27965927,   0.1, 0.0       ]]).astype(np.float32)

        matrix_mul = self.projection * self.view * self.model * np.transpose(buffer1)
        print(matrix_mul)

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
        # self.projection = ortho(-200, 200, -200, 200, -100, 100)
        self.program['u_projection'] = self.projection

    # ---------------------------------
    def on_mouse_wheel(self, event):
        self.translate += event.delta[1]
        self.translate = max(2, self.translate)
        self.view = translate((0, 0, -self.translate))
        self.program['u_view'] = self.view
        self.update()

    # ---------------------------------
    def on_draw(self, event):
        self.context.clear()
        self.program.draw('points')


if __name__ == '__main__':
    buffer = np.array([[ 64.59411621,  40.28605652,   0.1       ],
 [ 64.59389496,  40.28595352,   0.1       ],
 [ 64.59381104,  40.28591156,   0.1       ],
 [ 64.59369659,  40.28584671,   0.1       ],
 [ 64.59358978,  40.28577423,   0.1       ],
 [ 64.59277344,  40.28533173,   0.1       ],
 [ 64.59257507,  40.2852211 ,   0.1       ],
 [ 64.59215546,  40.2849884 ,   0.1       ],
 [ 64.59153748,  40.2846489 ,   0.1       ],
 [ 64.59139252,  40.28466415,   0.1       ],
 [ 64.59128571,  40.28461456,   0.1       ],
 [ 64.59088135,  40.28437805,   0.1       ],
 [ 64.59063721,  40.28423691,   0.1       ],
 [ 64.59036255,  40.28408813,   0.1       ],
 [ 64.59024811,  40.28401947,   0.1       ],
 [ 64.59013367,  40.28395844,   0.1       ],
 [ 64.58982086,  40.2837944 ,   0.1       ],
 [ 64.58907318,  40.28339005,   0.1       ],
 [ 64.58789825,  40.28277206,   0.1       ],
 [ 64.5867157 ,  40.28213882,   0.1       ],
 [ 64.58557892,  40.2815094 ,   0.1       ],
 [ 64.58435822,  40.28086472,   0.1       ],
 [ 64.58195496,  40.27965927,   0.1       ]]).astype(np.float32)
    c = Canvas(buffer)
    app.run()