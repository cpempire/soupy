from __future__ import absolute_import, division, print_function

from .costFunctional import CostFunctional
from .taylor import *
from .saa import *
# from .controlPDEProblem import ControlPDEProblem
# from .costFunctionalConstant import CostFunctionalConstant
# from .costFunctionalLinear import CostFunctionalLinear
# from .costFunctionalQuadratic import CostFunctionalQuadratic


# to do list
# 0. implement zero, Hessian term
# 1. implement linear
# 2. implement quadratic
# 3. impelement SAA

# to do list
# 1. SAA does not run well in ccgo1, multiprocessor does not work,
### not clear bug, simplifing adjoint solver works
# 2. quadratic approximation does not converge well, even without variance, does not converge
### record eigenvector after m_tr[i].zero()
# 3. check gradient for quadratic + correction

# what to show tomorrow
# 1. variance reduction by mean square error
# 2. trace estimation by MC and randomized SVD
# 3. scaling with repsect to mesh (design + uncertainty), trace, variance reduction, #bfgs
# 4. show the design and state, for both disk and submarine
# 5. random sample and state at different design


# April 9, 2018, work on reporting results
# 1. random samples and states at different design
# 2. table for variance reduction
# 3. plot trace estimation
# 4. plot #bfgs iterations
# obtain all results as planned