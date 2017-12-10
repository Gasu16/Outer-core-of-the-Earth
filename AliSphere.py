from mayavi import mlab
import numpy as np

sphere = mlab.points3d(0, 0, 0, scale_mode='none',
                                scale_factor=2,
                                color=(0.22, 0.77, 0.123),
                                resolution=100,
                                opacity=1,
                                name='Earth')

mlab.view(63.4, 73.8, 4, [-0.05, 0, 0])
mlab.show()