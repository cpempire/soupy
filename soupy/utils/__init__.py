# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.


from __future__ import absolute_import, division, print_function

# common
from .linalg import MatMatMult, MatPtAP, MatAtB, Transpose, \
                   to_dense, trace, get_diagonal, estimate_diagonal_inv2,  \
                   amg_method, DiagonalOperator, Solver2Operator, Operator2Solver
from .multivector import MultiVector, MatMvMult, MvDSmatMult
from .petsc4py_interface import make_operator, setup_preconditioner
from .variables import STATE, PARAMETER, ADJOINT, OPTIMIZATION
from .vector2function import vector2Function
from .checkDolfinVersion import dlversion
from .parameterList import ParameterList

# hIPPYlib algorithms
from .randomizedEigensolver import singlePass, doublePass, singlePassG, doublePassG
from .lowRankOperator import LowRankOperator
from .traceEstimator import TraceEstimator
from .cgsampler import CGSampler
from .random import Random
