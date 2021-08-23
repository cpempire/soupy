from __future__ import absolute_import, division, print_function

import pickle
import numpy as np
import dolfin as dl
import time
dl.dx = dl.dx(metadata={'quadrature_degree':2, "representation":'uflacs'}) #, "representation":'uflacs'

# import sys
# sys.path.append('../')
from ..costFunctional import CostFunctional
# sys.path.append('../../')
from ...utils.variables import STATE, PARAMETER, ADJOINT, OPTIMIZATION
from ...utils.vector2function import vector2Function
from ...utils.random import Random
from ...utils.multivector import MultiVector
from ...utils.randomizedEigensolver import doublePassG
from ...utils.checkDolfinVersion import dlversion
from ...models.reducedHessianSVD import ReducedHessianSVD


class ChanceConstraint(object):
    """
    an abstract class to compute the cost functional with objective and chance constraint, as well as its derivatives
    """

    def __init__(self, parameter, Vh, pde, qoi, qoi_constraint, prior, penalization, tol=1e-9):

        self.parameter = parameter
        self.Vh = Vh
        self.pde = pde
        self.generate_optimization = pde.generate_optimization
        self.qoi = qoi
        self.qoi_constraint = qoi_constraint
        self.prior = prior
        self.penalization = penalization
        self.z = pde.generate_optimization()
        self.m = prior.mean
        self.tol = tol

    def costValue(self, z):

        pass

    def costGradient(self, z):

        pass