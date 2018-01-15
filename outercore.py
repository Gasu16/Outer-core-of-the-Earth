import numpy


from pysph.base.utils import (get_particle_array_wcsph, get_particle_array_rigid_body)
from pysph.base.kernels import CubicSpline, WendlandQuintic
from pysph.sph.equation import Group

from pysph.sph.integrator import EPECIntegrator

from pysph.sph.integrator_step import WCSPHStep

from pysph.solver.application import Application

from pysph.solver.solver import Solver

from pysph.sph.rigid_body import RigidBodyMoments, RigidBodyMotion, RK2StepRigidBody

from pysph.sph.basic_equations import (XSPHCorrection, ContinuityEquation)

from pysph.sph.wc.basic import TaitEOSHGCorrection, MomentumEquation






# Parameters
dx = dy = dz = 0.0006  # m
hdx = 0.5
h = hdx * dx
r = 0.005
r1 = 0.008
r2 = 0.011

######################################################################
# Material properties: Table (1) of "A Free Lagrange Augmented Godunov Method
# for the Simulation of Elastic-Plastic Solids", B. P. Howell and G. J. Ball,
# JCP (2002)

# ALUMINIUM
ro1 = 2785.0  # refenrence density
C1 = 5328.0  # reference sound-speed
S1 = 1.338  # Particle-shock velocity slope
gamma1 = 2.0  # Gruneisen gamma/parameter
G1 = 2.76e7  # Shear Modulus (kPa)
Yo1 = 0.3e6  # Yield stress
E1 = ro1 * C1 * C1  # Youngs Modulus

# STEEL
ro2 = 7900.0  # reference density
C2 = 4600.0  # reference sound-speed
S2 = 1.490  # Particle shock velocity slope
gamma2 = 2.17  # Gruneisen gamma/parameter
G2 = 8.530e7  # Shear modulus
Yo2 = 0.979e6  # Yield stress
E2 = ro2 * C2 * C2  # Youngs modulus

# general
v_s = 3100.0  # inner_core velocity 3.1 km/s
cs1 = numpy.sqrt(E1 / ro1)  # speed of sound in aluminium
cs2 = numpy.sqrt(E2 / ro2)  # speed of sound in steel

######################################################################
# SPH constants and parameters

# Monaghan-type artificial viscosity
avisc_alpha = 1.0;
avisc_beta = 1.5;
avisc_eta = 0.1

# XSPH epsilon
xsph_eps = 0.5

# SAV1 artificial viscosity coefficients
alpha1 = 1.0
beta1 = 1.5
eta = 0.1  # in piab equation eta2 was written so final value is 0.01.(as req.)

# SAV2
alpha2 = 2.5
beta2 = 2.5
eta = 0.1  # in piab equation eta2 was written so final value is 0.01.(as req.)

# XSPH
eps = 0.5


######################################################################
# Particle creation rouintes
def get_inner_core_particles():
    x, y, z = numpy.mgrid[-r:r:dx, -r:r:dx, -r:r:dx] # -r is the start point; r is the end point of the vector; dx is step

    # We get 1-D array using ravel() function
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    d = (x * x + y * y + z * z)
    keep = numpy.flatnonzero(d <= r * r) # this function returns the index of non-zero elements of the array
    x = x[keep]
    y = y[keep]
    z = z[keep]

    print('%d inner_core particles' % len(x))

    hf = numpy.ones_like(x) * h
    mf = numpy.ones_like(x) * dx * dy * dz * ro2
    rhof = numpy.ones_like(x) * ro2
    csf = numpy.ones_like(x) * cs2
    u = numpy.ones_like(x) * v_s

    inner_core = get_particle_array_rigid_body(
        name="inner_core", x=x, y=y, z=z, h=hf, m=mf, rho=rhof, cs=csf, u=u)



    return inner_core


def get_outer_core_particles():
    x, y, z = numpy.mgrid[-r1:r1:dx, -r1:r1:dx, -r1:r1:dx]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    d = (x * x + y * y + z * z)
    keep = numpy.flatnonzero((d < r1 * r1) * (r * r < d))
    x = x[keep]
    y = y[keep]
    z = z[keep]


    print('%d Target particles' % len(x))


    hf = numpy.ones_like(x) * h
    mf = numpy.ones_like(x) * dx * dy * dz * ro1
    rhof = numpy.ones_like(x) * ro1
    csf = numpy.ones_like(x) * cs1
    u = numpy.ones_like(x)*0
    v = numpy.ones_like(x)*0
    w = numpy.ones_like(x)*0

    outer_core = get_particle_array_wcsph(name="outer_core", x=x, y=y, z=z, h=hf, m=mf, rho0=rhof,uo=u, v0=v, w0=w )
    return outer_core



def get_mantle():
    x, y, z = numpy.mgrid[-r2:r2:dx, -r2:r2:dx, -r2:r2:dx]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    d = (x * x + y * y + z * z)

    keep = numpy.flatnonzero((d <= r2 * r2) * (r1 * r1 <= d))
    x = x[keep]
    y = y[keep]
    z = z[keep]

    print('%d Target particles' % len(x))



    hf = numpy.ones_like(x) * h
    mf = numpy.ones_like(x) * dx * dy * dz * ro1
    rhof = numpy.ones_like(x) * ro1

    mantle = get_particle_array_rigid_body(name="mantle", x=x, y=y, z=z, h=hf, m=mf, rho=rhof)

    mantle.omega[2]= 5000000.0


    return mantle



class Impact(Application):
    def create_particles(self):
        outer_core = get_outer_core_particles()
        inner_core = get_inner_core_particles()
        mantle = get_mantle()

        return [outer_core, inner_core, mantle]

    def create_solver(self):
        dim = 3
        kernel = CubicSpline(dim=dim)
        #kernel = WendlandQuintic(dim=dim)

        self.wdeltap = kernel.kernel(rij=dx, h=hdx * dx)

        integrator = EPECIntegrator(inner_core=RK2StepRigidBody(), outer_core=WCSPHStep(), mantle=RK2StepRigidBody())
        solver = Solver(kernel=kernel, dim=dim, integrator=integrator)

        dt = 1e-9
        tf = 8e-6
        solver.set_time_step(dt)
        solver.set_final_time(tf)
        solver.set_print_freq(100)
        return solver

    def create_equations(self):
        equations = [
            Group(equations=[RigidBodyMoments(dest='inner_core', sources=None)]),

            #Group(equations=[RigidBodyMoments(dest='outer_core', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='inner_core', sources=None)]),
            #Group(equations=[RigidBodyMotion(dest='outer_core', sources=None)]),
            Group(equations=[RigidBodyMoments(dest='mantle', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='mantle', sources=None)]),





        ]
        return equations


if __name__ == '__main__':
    app = Impact()
app.run()