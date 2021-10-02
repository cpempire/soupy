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
incorporate advanced algorithms and capabilities, including 
-PDE-constrained tensor products,
-symbolic differentiation (of appropriate Lagrangians) for the
derivation of high order mixed derivatives (via the FEniCS interface),
-randomized algorithms for matrix and high order tensor decomposition,
-decomposition of uncertain parameter spaces by mixture models,
-product convolution approximations, 
-Taylor expansion-based high-dimensional control variates, and 
-inexact trust region preconditioned Newton-CG optimization methods. 

In addition, the library incorporates
common interfaces for random fields, PDE models, probability/risk
measures, and control/design/inversion constraints. 

The numerical
optimization algorithms will build on those implemented
in \textbf{SciPy} (\url{https://www.scipy.org/}), a Python-based
ecosystem of open-source software that includes a basic linear algebra
library implementing linear system and eigenvalue solvers, matrix
factorization, and an optimization library implementing such methods
as limited-memory BFGS, line search and trust region Newton-CG
algorithms, and others. These optimization methods are a good place to
start; however, they are not aimed at PDE-constrained optimization
problems and therefore do not respect the underlying
infinite-dimensional nature of the optimization problem, including
function spaces-aware norms, spectrally-equivalent preconditioners for
Hessians, regularizations for Lagrange multipliers for pointwise
inequality control and state constraints, and recognition of the
hierarchy of discretizations. As a result they generally do not
deliver mesh-independent convergence. The optimization algorithms in
hIPPYlib {\em are} infinite dimension-aware, and will be employed to
extend the SciPy algorithms for incorporation into SOUPy.
