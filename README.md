# soupy
Stochastic Optimization under high-dimensional Uncertainty in Python

![Alt text](SOUPyFramework.png?raw=true "Title")

SOUPy implements scalable algorithms to solve problems of PDE-constrained optimization under uncertainty, with the computational complexity measured in terms of PDE solves independent of the uncertain parameter dimension and optimization variable dimention. 

SOUPy is built on the open-source [hIPPYlib](https://hippylib.github.io/)
library, which provides state-of-the-art
scalable adjoint-based methods for deterministic and Bayesian inverse
problems governed by PDEs, which in turn makes use of the
[FEniCS](https://fenicsproject.org/) library for
high-level formulation, discretization, and scalable solution of
PDEs. 

SOUPy has been developed and in active development to
incorporate advanced approximation algorithms and capabilities, including 
- PDE-constrained operator/tensor/matrix products,
- symbolic differentiation (of appropriate Lagrangians) for the
derivation of high order mixed derivatives (via the FEniCS interface),
- randomized algorithms for matrix and high order tensor decomposition,
- decomposition of uncertain parameter spaces by mixture models,
- Taylor expansion-based high-dimensional control variates, and 
- product convolution approximations, 
- common interfaces for random fields, PDE models, probability/risk
measures, and control/design/inversion constraints. 

Numerical optimization algorithms can be called from [SciPy](https://www.scipy.org/) if used for low-dimensional optimization in serial computation. For high-dimensional optimization with large-scale PDE model solved by parallel computation, we provide distributed parallel optimization algorithms including 
- limited-memory BFGS with bound constraints, 
- line search or trust region inexact Newton-CG, 

which respect the underlying infinite-dimensional nature of the optimization problem, including
function spaces-aware norms, spectrally-equivalent preconditioners for
Hessians, regularizations for Lagrange multipliers for pointwise
inequality control and state constraints, and recognition of the
hierarchy of discretizations. 
