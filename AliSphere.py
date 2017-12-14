from mayavi import mlab
import numpy

# SPH equations
from pysph.sph.equation import Group

from pysph.sph.basic_equations import IsothermalEOS, ContinuityEquation, MonaghanArtificialViscosity, \
    XSPHCorrection, VelocityGradient3D

from pysph.sph.solid_mech.basic import MomentumEquationWithStress, HookesDeviatoricStressRate, \
    MonaghanArtificialStress, EnergyEquationWithStress

from pysph.sph.solid_mech.hvi import VonMisesPlasticity2D, MieGruneisenEOS, StiffenedGasEOS

from pysph.sph.gas_dynamics.basic import UpdateSmoothingLengthFromVolume

from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian, CubicSpline, WendlandQuintic
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator, EPECIntegrator
from pysph.sph.integrator_step import SolidMechStep

class Earth:
    # Set Earth parameters
    def __init__(self, ro1, C1, S1, gamma1, G1, Yo1):

        # type: (object, object, object, object, object, object) -> object

        self.ro1 = ro1
        self.C1 = C1
        self.S1 = S1
        self.gamma1 = gamma1
        self.G1 = G1
        self.Yo1 = Yo1


    def print_E1(self, ro1, C1):
        E1 = ro1 * C1 * C1
        print "E1: " + str(E1)

    def print_volume(self):
        radius = 3
        r2 = numpy.arange(-radius, radius+1)**2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        volume = numpy.sum(dist2 <= radius ** 2)
        print "Volume: " + str(volume)
        return volume


class Asteroid:
    def __init__(self):
        # TO DO CLASS
        pass



class Impact:
    def __init__(self, E = Earth(2785.0, 5328.0, 1.338, 2.0, 2.76e7, 0.3e6)):
        E.print_E1(E.ro1, E.C1)
        E.print_volume()



if __name__ == '__main__':
    app = Impact()
    #app.__init__(E = Earth(2785.0, 5328.0, 1.338, 2.0, 2.76e7, 0.3e6))

"""
sphere = mlab.points3d(0, 0, 0, scale_mode='none',
                                scale_factor=2,
                                color=(0.22, 0.77, 0.123),
                                resolution=100,
                                opacity=1,
                                name='Earth')

mlab.view(63.4, 73.8, 4, [-0.05, 0, 0])
mlab.show()
"""