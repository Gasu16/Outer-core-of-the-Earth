"""Three-dimensional dam break over a dry bed. (14 hours)

The case is described as a SPHERIC benchmark
 https://wiki.manchester.ac.uk/spheric/index.php/Test2

By default the simulation runs for 6 seconds of simulation time.
"""
import sys
from sys import settrace
from pysph.base.utils import (get_particle_array_wcsph, get_particle_array_rigid_body)
from pysph.base.kernels import CubicSpline, WendlandQuintic
from pysph.base.kernels import WendlandQuintic
from pysph.examples._db_geometry import DamBreak3DGeometry
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.scheme import WCSPHScheme
from pysph.sph.equation import Group
from pysph.sph.basic_equations import XSPHCorrection, ContinuityEquation
from pysph.sph.wc.basic import TaitEOS, MomentumEquation
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.rigid_body import RigidBodyMoments, RigidBodyMotion, RK2StepRigidBody, LiuFluidForce

dim = 3

dt = 1e-5
tf = 6.0

# parameter to change the resolution
dx = 0.1
nboundary_layers=3
hdx = 1.2
ro = 1000.0
h0 = dx * hdx
gamma = 7.0
alpha = 0.5
beta = 0.0

class DamB3D(Application):

    def initialize(self):
        self.geom = DamBreak3DGeometry(
            dx=dx, nboundary_layers=nboundary_layers, hdx=hdx, rho0=ro
        )
        self.co = 10.0 * self.geom.get_max_speed(g=9.81)

    def create_solver(self):
        dim = 3
        kernel = CubicSpline(dim=dim)
        # kernel = WendlandQuintic(dim=dim)

        self.wdeltap = kernel.kernel(rij=dx, h=hdx * dx)

        integrator = EPECIntegrator(fluid=WCSPHStep())
        solver = Solver(kernel=kernel, dim=dim, integrator=integrator)

        dt = 1e-9
        tf = 8e-6
        solver.set_time_step(dt)
        solver.set_final_time(tf)
        solver.set_print_freq(100)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[
                LiuFluidForce(dest='fluid', sources=None, ),
                XSPHCorrection(dest='fluid', sources=['fluid', ]),
                TaitEOS(dest='fluid', sources=None, rho0=1000,
                        c0=1498, gamma=7.0),
            ], real=False),

            Group(equations=[
                ContinuityEquation(dest='fluid', sources=['fluid', ]),

                MomentumEquation(dest='fluid', sources=['fluid'],
                                 alpha=0.1, beta=0.0, c0=1498, gy=-9.81),

                XSPHCorrection(dest='fluid', sources=['fluid']),

            ]),
        ]

        return equations


    def create_particles(self):
        return self.geom.create_particles()


if __name__ == '__main__':
    app = DamB3D()
    app.run()

