from __future__ import absolute_import, division, print_function

import math
import numpy as np
import dolfin as dl
dl.dx = dl.dx(metadata={'quadrature_degree':2, "representation":'uflacs'})
dl.set_log_active(False)

# import mshr as mh
import pickle
from qoiHelmholtz import QoIHelmholtz

# path = ""
path = "../../../"

import sys
sys.path.append(path)
from soupy import *

# # samples used to compute mean square error
N_mse = 10

# number of PDEs
N_pde = 16

# choose mesh size, [1, 2, 3, 4, 5]
meshsize = 1

geometry = 'square_full'
filename = "../mesh/"+geometry+str(meshsize)+".xml"
mesh = dl.Mesh(filename)

optMethod = 'home_bfgs'
# optMethod = 'home_ncg'
# optMethod = 'scipy_bfgs'
# optMethod = 'fmin_ncg'

if optMethod is 'home_bfgs':
    bfgs_ParameterList = BFGS_ParameterList()
    bfgs_ParameterList["max_iter"] = 32
    bfgs_ParameterList["LS"]["max_backtracking_iter"] = 10
    bfgs_ParameterList["BFGS_op"]["memory_limit"] = 10
elif optMethod is 'home_ncg':
    ncg_ParameterList = ReducedSpaceNewtonCG_ParameterList()
    ncg_ParameterList["globalization"] = "LS"
    ncg_ParameterList["max_iter"] = 16
    ncg_ParameterList["LS"]["max_backtracking_iter"] = 20

plotFlag = False

############ 0 parameters to config ###############

## use Monte Carlo correction True/False
correction = False

# run the constant approximation True/False
constant_run = True
# run the linear approximation True/False
linear_run = True
# run the quadratic approximation True/False
quadratic_run = True
# run the SAA approximation True/False
saa_run = True

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
    # geometry
    parameter["geometry"] = geometry
    # mesh size
    parameter["meshsize"] = meshsize
    # number of samples for trace estimator
    parameter['N_tr'] = 50
    # number of samples for Monte Carlo correction
    parameter['N_mc'] = 10
    # regularization coefficient
    parameter['alpha'] = 1.e-2  # alpha*R(z)
    # variance coefficient
    parameter['beta'] = 1.0     # E[Q] + beta Var[Q]
    # prior covariance, correlation length prop to gamma/delta
    parameter['delta'] = 1.    # delta*Identity
    parameter['gamma'] = 0.5    # gamma*Laplace
    parameter['dim'] = 1  # number of optimization variables

    pickle.dump(parameter, open('data/parameter.p','wb'))

################### 1. Define the Geometry ###########################

thickness = 1.
r1, r2 = 1., 3.
cx, cy = 5., 5.
Lx, Ly = 2*cx, 2*cy
Lx_minus = 0 #-thickness
Lx_plus = Lx + thickness
Ly_minus = -thickness
Ly_plus = Ly + thickness

comm = mesh.mpi_comm()
mpi_comm = mesh.mpi_comm()
if hasattr(mpi_comm, "rank"):
    mpi_rank = mpi_comm.rank
    mpi_size = mpi_comm.size
else:
    mpi_rank = 0
    mpi_size = 1

class Domain:
    # the domain of the acoustic wave propagation
    def __init__(self, mesh):

        self.r1, self.r2 = 1.0, 3.0  # inner and outer radius
        self.cx, self.cy = 5.0, 5.0  # center (cx, cy)
        domain = [0.0, 0.0, 10.0, 10.0]  # domain [0, 0, Lx, Ly]

        # homogeneous medium
        medium = dl.AutoSubDomain(
            lambda x, on_boundary: x[0] >= 0.0 and x[0] <= 10.0 and x[1] >= 0.0 and x[1] <= 10.0)

        # PML layer
        thickness = 1.0  # PML thickness
        self.sigma_x = dl.Expression("(x[0]<xL)*A*(x[0]-xL)*(x[0]-xL)/(t*t) + (x[0]>xR)*A*(x[0]-xR)*(x[0]-xR)/(t*t)",
                                xL=domain[0], xR=domain[2], A=50, t=thickness, degree=1)
        # sigma_x = dl.interpolate(sigma_x, Vh_PARAMETER)

        self.sigma_y = dl.Expression("(x[1]<yB)*A*(x[1]-yB)*(x[1]-yB)/(t*t) + (x[1]>yT)*A*(x[1]-yT)*(x[1]-yT)/(t*t)",
                                yB=domain[1], yT=domain[3], A=50, t=thickness, degree=1)
        # sigma_y = dl.interpolate(sigma_y, Vh_PARAMETER)

        # design region
        design = dl.AutoSubDomain(lambda x, on_boundary:
                                  (pow(x[0]-5.0, 2)+pow(x[1]-5.0, 2) >= 1.0) and (pow(x[0]-5.0, 2)+pow(x[1]-5.0, 2) <= 9.0))

        cell_marker = dl.CellFunction("size_t", mesh)
        cell_marker.set_all(0)
        medium.mark(cell_marker, 1)
        design.mark(cell_marker, 2)
        self.dx = dl.Measure("dx", subdomain_data=cell_marker)
        self.dx = self.dx(metadata={'quadrature_degree': 2, "representation": 'uflacs'})

domain = Domain(mesh)

#################### 2. Define control PDE problem #######################
dim = 3  # number of optimization variables is dim
parameter["dim"] = dim
Vh_STATE = dl.VectorFunctionSpace(mesh, 'CG', 1)
Vh_STATE_sub = dl.FunctionSpace(mesh, 'CG', 1)
Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
Vh_OPTIMIZATION = dl.VectorFunctionSpace(mesh, "DG", 0, dim=dim)
Vh_OPTIMIZATION_sub = dl.FunctionSpace(mesh, "DG", 0)
Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_OPTIMIZATION]

chi_medium = dl.Expression("( ( pow(x[0]-cx,2)+pow(x[1]-cy,2) ) >= pow(r1,2) )", cx=cx, cy=cy, r1=r1, degree=1)
chi_medium = dl.interpolate(chi_medium, Vh_OPTIMIZATION_sub)
chi_obstacle = dl.Expression("( ( pow(x[0]-cx,2)+pow(x[1]-cy,2) ) <= pow(r1,2) )", cx=cx, cy=cy, r1=r1, degree=1)
chi_obstacle = dl.interpolate(chi_obstacle, Vh_OPTIMIZATION_sub)
chi_observation = dl.Expression("( ( pow(x[0]-cx,2)+pow(x[1]-cy,2) ) > pow(r2,2) )", cx=cx, cy=cy, r2=r2, degree=1)
chi_observation = dl.interpolate(chi_observation, Vh_OPTIMIZATION_sub)
chi_design = dl.Expression("((pow(x[0]-cx,2)+pow(x[1]-cy,2))>=pow(r1,2))*((pow(x[0]-cx,2)+pow(x[1]-cy,2))<=pow(r2,2))",
                           cx=cx, cy=cy, r1=r1, r2=r2, degree=1)
chi_design = dl.interpolate(chi_design, Vh_OPTIMIZATION_sub)

sigma_x = dl.Expression("(x[0]<xL)*A*(x[0]-xL)*(x[0]-xL)/(t*t) + (x[0]>xR)*A*(x[0]-xR)*(x[0]-xR)/(t*t)",
                        xL=0., xR=Lx, A=50, t=thickness, degree=1)
sigma_x = dl.interpolate(sigma_x, Vh_PARAMETER)

sigma_y = dl.Expression("(x[1]<yB)*A*(x[1]-yB)*(x[1]-yB)/(t*t) + (x[1]>yT)*A*(x[1]-yT)*(x[1]-yT)/(t*t)",
                        yB=0., yT=Ly, A=50, t=thickness, degree=1)
sigma_y = dl.interpolate(sigma_y, Vh_PARAMETER)


def residual(u, m, p, z, data=None):
    # reference: E. Turkel, A. Yefet, Absorbing PML boundary layers for wave-like equations, 1998, Applied Numerical Mathematics

    if data is not None:
        frequency = data['frequency']
        alpha = data['alpha']
        px = 5.-4.*math.cos(alpha)
        py = 5.-4.*math.sin(alpha)
        pointsouce = dl.Expression("10./h*exp(-( pow(x[0]-px,2)+pow(x[1]-py,2) )/h)", px=px, py=py, h=0.1, degree=2)

    omega = 2.*math.pi*frequency

    k0 = dl.Constant(omega/c_air)

    sigma_x, sigma_y = domain.sigma_x, domain.sigma_y
    ur, ui = dl.split(u)
    pr, pi = dl.split(p)
    z1, z2, z3 = dl.split(z)

    # print(frequency)

    k = k0 * dl.exp(dl.exp(m) * z1 * chi_design)

    # single design
    rhoInv = dl.Constant(1.0)

    # # double design
    # rhoInv = dl.exp(dl.exp(m) * z2 * chi_design)

    # # triple design
    # rho11 = dl.exp(dl.exp(m) * z2 * chi_design)
    # rho12 = dl.exp(m) * z3 * chi_design
    # rhoInv = dl.as_matrix([[rho11, rho12],[rho12, rho11]])

    ksquared = k**2

    Kr = ksquared - sigma_x * sigma_y
    Ki = -k * (sigma_x + sigma_y)

    Dr_xx = (ksquared + sigma_x * sigma_y) / (ksquared + sigma_x * sigma_x)
    Dr_yy = (ksquared + sigma_x * sigma_y) / (ksquared + sigma_y * sigma_y)
    Di_xx = k * (sigma_x - sigma_y) / (ksquared + sigma_x * sigma_x)
    Di_yy = k * (sigma_y - sigma_x) / (ksquared + sigma_y * sigma_y)

    Dr = dl.as_matrix([[Dr_xx, dl.Constant(0.)], [dl.Constant(0.), Dr_yy]])
    Di = dl.as_matrix([[Di_xx, dl.Constant(0.)], [dl.Constant(0.), Di_yy]])

    form_pml_r = dl.inner(Dr * dl.grad(ur), dl.grad(pr)) * domain.dx(0) \
                 + dl.inner(Di * dl.grad(ui), dl.grad(pr)) * domain.dx(0) \
                 - Kr * ur * pr * domain.dx(0) \
                 - Ki * ui * pr * domain.dx(0)

    form_pml_i = - dl.inner(Dr * dl.grad(ui), dl.grad(pi)) * domain.dx(0) \
                 + dl.inner(Di * dl.grad(ur), dl.grad(pi)) * domain.dx(0) \
                 + Kr * ui * pi * domain.dx(0) \
                 - Ki * ur * pi * domain.dx(0)

    form_medium_r = dl.inner(dl.grad(ur), dl.grad(pr)) * domain.dx(1) \
                    - ksquared * ur * pr * domain.dx(1)

    form_medium_i = -dl.inner(dl.grad(ui), dl.grad(pi)) * domain.dx(1) \
                    + ksquared * ui * pi * domain.dx(1)

    form_design_r = dl.inner( rhoInv * dl.grad(ur), dl.grad(pr)) * domain.dx(2) \
                    - ksquared * ur * pr * domain.dx(2)

    form_design_i = -dl.inner( rhoInv * dl.grad(ui), dl.grad(pi)) * domain.dx(2) \
                    + ksquared * ui * pi * domain.dx(2)

    form_source = pointsouce * pr * domain.dx(1) - dl.Constant(0.0) * pi * domain.dx(1)

    # form_incident_r =  dl.inner( (rhoInv - dl.Constant(1.0)) * dl.grad(ur_incident), dl.grad(pr)) * domain.dx(2) \
    #                 - (ksquared - k0**2) * ur_incident * pr * domain.dx(2)
    #
    # form_incident_i = -dl.inner( (rhoInv - dl.Constant(1.0)) * dl.grad(ui_incident), dl.grad(pi)) * domain.dx(2)\
    #                 + (ksquared - k0**2) * ui_incident * pi * domain.dx(2)


    form = form_pml_r + form_pml_i + form_medium_r + form_medium_i +  \
           form_design_r + form_design_i + form_source  # form_incident_r + form_incident_i

    return form


def boundary_obstacle(x, on_boundary):
    return dl.near((x[0]-cx)**2+(x[1]-cy)**2, r1**2, 1e-8)   # and on_boundary

c_air = 343.4        # m/s
# frequency_set = np.ones(N_pde)*c_air #np.linspace(300, 600, N_pde) #
frequency_set = np.linspace(c_air/2, c_air, N_pde)
alpha_set = np.array(range(N_pde))/N_pde*2*math.pi  # np.linspace(0, 2*math.pi, N_pde)
# alpha_set = np.zeros(N_pde)
pde = []
u_incident = []
for i in range(N_pde):

    frequency = frequency_set[i]
    alpha = alpha_set[i]

    data = dict()
    data['frequency'] = frequency
    data['alpha'] = alpha

    bc1 = dl.DirichletBC(Vh_STATE.sub(0), dl.Constant(0.0), boundary_obstacle,
                         method='topological', check_midpoint=False)
    bc2 = dl.DirichletBC(Vh_STATE.sub(1), dl.Constant(0.0), boundary_obstacle,
                         method='topological', check_midpoint=False)
    bcs = [bc1, bc2]

    bc1 = dl.DirichletBC(Vh_STATE.sub(0), dl.Constant(0.0), boundary_obstacle,
                         method='topological', check_midpoint=False)
    bc2 = dl.DirichletBC(Vh_STATE.sub(1), dl.Constant(0.0), boundary_obstacle,
                         method='topological', check_midpoint=False)
    bcs0 = [bc1, bc2]

    pde.append(ControlPDEProblemMultiPDE(Vh, residual, bcs, bcs0, is_fwd_linear=True, data=data))

################## 3. Define the quantity of interest (QoI) ############
qoi = []
for i in range(N_pde):
    u_obs = dl.Function(Vh_STATE, name="state")

    frequency = frequency_set[i]
    alpha = alpha_set[i]

    data = dict()
    data['frequency'] = frequency
    data['alpha'] = alpha

    m = dl.Function(Vh_PARAMETER)
    p = dl.TestFunction(Vh_STATE)
    z = dl.Function(Vh_OPTIMIZATION)
    res_form = residual(u_obs, m, p, z, data=data)
    dl.solve(res_form == 0, u_obs, bcs=[])

    ur_obs, ui_obs = u_obs.split()

    ur_obs, ui_obs = u_obs.split()

    if plotFlag:
        dl.plot(ur_obs, title='observed state real')
        dl.plot(ui_obs, title='observed state imag')

    if dlversion() <= (1, 6, 0):
        dl.File(comm, "data/ur_obs_"+str(i)+".xdmf") << ur_obs
        dl.File(comm, "data/ui_obs_"+str(i)+".xdmf") << ui_obs
    else:
        xf = dl.XDMFFile(comm, "data/ur_obs_"+str(i)+".xdmf")
        xf.write(ur_obs)
        xf = dl.XDMFFile(comm, "data/ui_obs_"+str(i)+".xdmf")
        xf.write(ui_obs)

    qoi.append(QoIHelmholtz(mesh, Vh_STATE, chi_observation, u_obs))

################## 4. Define Penalization term ############################

penalization = L1PenalizationMultiVariable(Vh[OPTIMIZATION], dl.dx, parameter["alpha"], region=chi_design, dim=dim)

################## 5. Define the prior ##################### ##############
prior = BiLaplacianPrior(Vh[PARAMETER], parameter["gamma"], parameter["delta"])
