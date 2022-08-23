import random as rd
from dolfin import *

import numpy as np
path = "../../"
import sys
sys.path.append(path)
from soupy import *

from model_ok_min_hippylib import *
from cahn_hilliard import CahnHilliardProblem
from fullNewtonSolver import FullNewtonSolver

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        # rd.seed(4)
        if has_pybind11():
            super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0. + 1.*(0.5 - rd.random())
        values[1] = 0.0
    def value_shape(self):
        return (2,)

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and build function space
# mesh = UnitSquareMesh.create(100, 100, CellType.Type_quadrilateral)
mesh = UnitSquareMesh(100, 100)
# mesh = UnitCubeMesh(32, 32, 32)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Vh = FunctionSpace(mesh, P1*P1)

# Define functions
z   = Function(Vh)  # current solution
z0  = Function(Vh)  # solution from previous converged step


Ntrials = 1
iter_count = np.zeros((Ntrials, 3))

# parameters
tol = 1e-6
kappa = 1.  # double well potential
epsilon = 0.01  # interface
sigma = 500.  # nonlocal term
theta = np.array([kappa, epsilon, sigma])
for iter in range(Ntrials):
    # Create intial conditions and interpolate
    z_init = InitialConditions(degree=1)
    z0.interpolate(z_init)
    # z1 = z0
    z1 = project(z0, Vh)
    # assign(z0.sub(0),sample_fun)

    z.assign(z1)
    m = assemble(z1.split()[0] * dx)
    #########################################################
    if True:
        # solver with reduced Newton formulation, Hessian modification, and line search
        solver = ok_min(Vh, m, theta, save=True)
        state = Function(solver.Vh[STATE])
        x = [Function(V) for V in solver.Vh]
        x[PARAMETER].vector().set_local(np.array([theta[0]/4, theta[1], theta[2]]))
        solver.solveFwd(state, x, z1, tol=tol)

        iter_count[iter, 0] = solver.it
    ###########################################
    if True:
        # solver with full Newton formulation, no Hessian modification, no line search (optional)
        problem = CahnHilliardProblem(Vh, m, theta, rescaling=False, save=True)
        solver = FullNewtonSolver(problem)
        solver.parameters["maximum_iterations"] = 500
        solver.parameters["rel_tolerance"] = tol
        z.assign(z1)

        mu = problem.mu_solve(z.vector())
        assign(z.sub(1), mu)
        result = solver.solve(z.vector())
        iter_count[iter, 1] = solver.it
        ###########################################
    if True:
        # solver with rescaled full Newton formulation, and Hessian modification, no line search (optional)
        problem = CahnHilliardProblem(Vh, m, theta, rescaling=True, save=True)
        solver = FullNewtonSolver(problem)
        solver.parameters["maximum_iterations"] = 500
        solver.parameters["rel_tolerance"] = tol
        z.assign(z1)

        mu = problem.mu_solve(z.vector())
        assign(z.sub(1), mu)
        result = solver.solve(z.vector())
        iter_count[iter, 2] = solver.it

    if True:
        # FEniCS self solver with full Newton formulation, no Hessian modification, no line search
        # Class for interfacing with the Newton solver
        class CahnHilliardEquation(NonlinearProblem):
            def __init__(self, a, L):
                NonlinearProblem.__init__(self)
                self.L = L
                self.a = a

            def F(self, b, x):
                assemble(self.L, tensor=b)

            def J(self, A, x):
                assemble(self.a, tensor=A)

        # Define trial and test functions
        dz    = TrialFunction(Vh)
        v, nu  = TestFunctions(Vh)

        # Split mixed functions
        u,  mu  = split(z)

        # Weak statement of the equations
        L0 = dot(grad(mu), grad(v))*dx + sigma*(u - m)*v*dx
        L1 = mu*nu*dx + kappa*(1-u*u)*u*nu*dx - epsilon*epsilon*dot(grad(u),grad(nu))*dx
        L = L0 + L1

        # Compute directional derivative about z in the direction of dz (Jacobian)
        a = derivative(L, z, dz)

        # Create nonlinear problem and Newton solver
        problem = CahnHilliardEquation(a, L)
        solver = NewtonSolver()
        # solver.set_relaxation_parameter(0.1)
        solver.parameters["linear_solver"] = "lu"
        solver.parameters["convergence_criterion"] = "incremental"
        solver.parameters["relative_tolerance"] = 1e-8
        solver.parameters["maximum_iterations"] = 500

        # Output file
        File("data/input-u.pvd") << z.split()[0]
        # File("data/input-mu.pvd") << z.split()[1]

        # solve the problem
        z.assign(z1)
        solver.solve(problem, z.vector())

        File("data/output-u.pvd") << z.split()[0]
        # File("data/output-mu.pvd") << z.split()[1]


np.save("iterations", iter_count)

    # # sampling from Laplace prior
# gamma = 1e2
# delta = 1e3
# prior_init = LaplacianPrior(Vhsub, gamma, delta)
# prior_init.mean.set_local(0.*np.ones(len(prior_init.mean.get_local())))
#
# noise = Vector()
# prior_init.init_vector(noise, "noise")
# noise.set_local(np.random.normal(0, 1, len(noise.get_local())))
# # randomGen = Random(nproc=mesh.mpi_comm().size)
# # randomGen.normal(1., noise)
#
# sample = Vector()
# prior_init.init_vector(sample, 1)
# prior_init.sample(noise, sample, add_mean=True)
# sample_fun = vector2Function(sample, Vhsub)
# File("data/sample.pvd") << sample_fun
#
# print("mean", assemble(sample_fun * dx))
