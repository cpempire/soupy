from __future__ import absolute_import, division, print_function

import math
import numpy as np
import dolfin as dl
dl.dx = dl.dx(metadata={'quadrature_degree':2, "representation":'uflacs'})
dl.set_log_active(False)

# import mshr as mh
import pickle
from qoiReservoir import QoIObjective, QoIConstraint

# path = ""
path = "../../"

import sys
sys.path.append(path)
from soupy import *

# # samples used to compute mean square error
N_mse = 10

# choose mesh size, [1, 2, 3, 4, 5]
meshsize = 2

# optMethod = 'home_bfgs'
# optMethod = 'home_ncg'
optMethod = 'scipy_bfgs'
# optMethod = 'fmin_ncg'

if optMethod is 'home_bfgs':
    bfgs_ParameterList = BFGS_ParameterList()
    bfgs_ParameterList["max_iter"] = 16
    bfgs_ParameterList["LS"]["max_backtracking_iter"] = 10
elif optMethod is 'home_ncg':
    ncg_ParameterList = ReducedSpaceNewtonCG_ParameterList()
    ncg_ParameterList["globalization"] = "LS"
    ncg_ParameterList["max_iter"] = 16
    ncg_ParameterList["LS"]["max_backtracking_iter"] = 10
elif optMethod is 'scipy_bfgs':
    from scipy.optimize import fmin_l_bfgs_b

plotFlag = False

############ 0 parameters to config ###############

## use Monte Carlo correction True/False
correction = False

active = [1, 1, 1, 1]
# active = [0, 0, 0, 1]
# active = [0, 0, 0, 1]

# run the constant approximation True/False
constant_run = active[0]
# run the linear approximation True/False
linear_run = active[1]
# run the quadratic approximation True/False
quadratic_run = active[2]
# run the SAA approximation True/False
saa_run = active[3]

# check gradient True/False
check_gradient = False
# check Hessian True/False
check_hessian = False

if correction:
    parameter = pickle.load(open('data/parameter.p', 'rb'))
    parameter["correction"] = correction
else:
    parameter = dict()
    # with Monte Carlo correction or not
    parameter["correction"] = correction
    # optimization method
    parameter["optMethod"] = optMethod
    # mesh size
    parameter["meshsize"] = meshsize
    parameter["optType"] = 'vector'
    parameter["optDimension"] = 25
    parameter["bounds"] = [(0., 32.) for i in range(parameter["optDimension"])]
    # number of eigenvalues for trace estimator
    parameter['N_tr'] = 10
    # number of samples for Monte Carlo correction
    parameter['N_mc'] = 100
    # regularization coefficient
    parameter['alpha'] = 1.e-5  # alpha*R(z)
    # variance coefficient
    parameter['beta'] = 0.     # E[Q] + beta Var[Q]
    # prior covariance, correlation length prop to gamma/delta
    parameter['delta'] = 5.    # delta*Identity
    parameter['gamma'] = 0.05    # gamma*Laplace

    # number of samples for Monte Carlo evaluation of chance/probability P(f \leq 0)
    parameter['N_cc'] = 1024
    # chance c_alpha in the chance constraint P(f \leq 0) \geq c_alpha
    parameter['c_alpha'] = 0.05
    # smoothing parameter of indicator function I_{(-\infty, 0]}(x) \approx 1 / ( 1 + \exp(-2 c_\beta x) )
    parameter['c_beta'] = 4.
    # penalty parameter of the inequality chance constraint c_gamma/2 (\max(0, x))^2
    parameter['c_gamma'] = 100.  # 100.

    pickle.dump(parameter, open('data/parameter.p','wb'))

################### 1. Define the Geometry ###########################

dl.parameters["ghost_mode"] = "shared_facet"

# 1. Define the Geometry
nx = 32
ny = 32
mesh = dl.RectangleMesh(dl.Point(0., 0.), dl.Point(1., 1.), nx, ny)

comm = mesh.mpi_comm()
mpi_comm = mesh.mpi_comm()
if hasattr(mpi_comm, "rank"):
    mpi_rank = mpi_comm.rank
    mpi_size = mpi_comm.size
else:
    mpi_rank = 0
    mpi_size = 1

#################### 2. Define optimization PDE problem #######################

Vh_STATE = dl.FunctionSpace(mesh, 'CG', 1)
Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
Vh_OPTIMIZATION = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=parameter["optDimension"])
Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_OPTIMIZATION]


# Define Dirichlet boundary (on all boundaries)
def boundary(x):
    return x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS or x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS
    # return x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS


u_bdr = dl.Expression("0.0", degree=2)
u_bdr0 = dl.Expression("0.0", degree=2)
bcs = dl.DirichletBC(Vh[STATE], u_bdr, boundary)
bcs0 = dl.DirichletBC(Vh[STATE], u_bdr0, boundary)

mollifierlist = []
n1d = np.int(np.sqrt(parameter["optDimension"]))
x = np.linspace(0.25, 0.75, n1d)
y = np.linspace(0.25, 0.75, n1d)
for i in range(n1d):
    for j in range(n1d):
        mollifierlist.append(dl.Expression("exp(-(pow(x[0]-xi,2)+pow(x[1]-yj,2))/(pow(0.1,2)))", xi=x[i], yj=y[j], degree=2))

mollifier = dl.as_vector(mollifierlist)


def residual(u, m, p, z):

    form = dl.exp(m)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx + dl.inner(mollifier, z)*p*dl.dx

    return form


pde = ControlPDEProblem(Vh, residual, bcs, bcs0, is_fwd_linear=True)

################## 3. Define the quantity of interest (QoI) ############

qoi_objective = QoIObjective(mesh, Vh)

qoi_constraint = QoIConstraint(mesh, Vh)

################## 4. Define Penalization term ############################

penalization = L2PenalizationFiniteDimension(Vh[OPTIMIZATION], dl.dx, parameter["alpha"])

################## 5. Define the prior ##################### ##############
anis_diff = dl.Expression(code_AnisTensor2D, degree=2)
anis_diff.theta0 = 1.
anis_diff.theta1 = 1.
anis_diff.alpha = 0.*np.pi/4

# if correction:
#     ############ begin load atrue  #####################
#     atrue_array = pickle.load( open( "data/atrue_array.p", "rb" ) )
#     atrue_fun = dl.Function(Vh_PARAMETER, name="sample")
#     atrue = atrue_fun.vector()
#     atrue[:] = atrue_array[:]
#     ############ finish load atrue #####################
# else:
#     ########## begin create atrue ####################
#     gamma_atrue = .01
#     delta_atrue = 4
#
#     def true_model(Vh, gamma, delta, anis_diff):
#         prior = BiLaplacianPrior(Vh, gamma, delta, anis_diff)
#         noise = dl.Vector()
#         prior.init_vector(noise,"noise")
#         noise_size = noise.get_local().shape[0]
#         noise_true = np.random.randn(noise_size)
#         np.save("data/noise_true",noise_true)
#         noise_true = np.load("data/noise_true.npy")
#         noise.set_local( noise_true)
#         # noise = dl.Vector()
#         # prior.init_vector(noise,"noise")
#         # Random.normal(noise, 1., True)
#         atrue = dl.Vector()
#         prior.init_vector(atrue, 0)
#         prior.sample(noise,atrue)
#         return atrue
#
#     atrue = true_model(Vh[PARAMETER], gamma_atrue, delta_atrue, anis_diff)
#     atrue_fun = vector2Function(atrue, Vh_PARAMETER, name="sample")
#     # dl.plot(atrue_fun)
#
#     locations = np.array([[.2, .1], [.2, .9], [.5,.5], [.8, .1], [.8, .9]])
#     pen = 2e1
#     gamma_moll = 1.
#     delta_moll = 1.
#     prior = MollifiedBiLaplacianPrior(Vh[PARAMETER], gamma_moll, delta_moll, locations, atrue, anis_diff, pen)
#     atrue = prior.mean
#     atrue_fun = vector2Function(atrue, Vh_PARAMETER, name="sample")
#     # dl.plot(atrue_fun)
#
#     atrue_array = np.array(atrue.array())
#     pickle.dump( atrue_array, open( "data/atrue_array.p", "wb" ) )
#     dl.File("data/atrue.pvd") << atrue_fun
#     ########### finish create atrue ####################
#
# prior = BiLaplacianPrior(Vh[PARAMETER], parameter["gamma"], parameter["delta"], anis_diff, atrue)

prior = BiLaplacianPrior(Vh[PARAMETER], parameter["gamma"], parameter["delta"], anis_diff)
