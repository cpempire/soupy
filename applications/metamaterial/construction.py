from __future__ import absolute_import, division, print_function

import math
import numpy as np
import dolfin as dl
dl.dx = dl.dx(metadata={'quadrature_degree': 3, "representation":'uflacs'})
dl.set_log_active(False)

# import mshr as mh
import pickle
from qoiHelmholtz import QoIHelmholtz

# path = ""
path = "../../"

import sys
sys.path.append(path)
from soupy import *

# # samples used to compute mean square error
N_mse = 10

# choose mesh size, [1, 2, 3, 4, 5]
meshsize = 1

geometry = 'disk'
filename = "mesh/"+geometry+str(meshsize)+".xml"
mesh = dl.Mesh(filename)

optMethod = 'home_bfgs'
# optMethod = 'home_ncg'
# optMethod = 'scipy_bfgs'
# optMethod = 'fmin_ncg'

if optMethod is 'home_bfgs':
    bfgs_ParameterList = BFGS_ParameterList()
    bfgs_ParameterList["max_iter"] = 32
    bfgs_ParameterList["LS"]["max_backtracking_iter"] = 10
elif optMethod is 'home_ncg':
    ncg_ParameterList = ReducedSpaceNewtonCG_ParameterList()
    ncg_ParameterList["globalization"] = "LS"
    ncg_ParameterList["max_iter"] = 32
    ncg_ParameterList["LS"]["max_backtracking_iter"] = 10

plotFlag = False

############ 0 parameters to config ###############

## use Monte Carlo correction True/False
correction = False

# run the constant approximation True/False
constant_run = False
# run the linear approximation True/False
linear_run = False
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
    parameter['delta'] = 50.    # delta*Identity
    parameter['gamma'] = 50.    # gamma*Laplace

    pickle.dump(parameter, open('data/parameter.p','wb'))

################### 1. Define the Geometry ###########################

thickness = 1.

if geometry is 'disk':
    Lx = 20
    Ly = 10
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

#################### 2. Define control PDE problem #######################

Vh_STATE = dl.VectorFunctionSpace(mesh, 'CG', 2)
Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
Vh_OPTIMIZATION = dl.FunctionSpace(mesh, "CG", 1)
Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_OPTIMIZATION]

chi_medium = dl.Expression("( ( pow(x[0]-10.,2)+pow(x[1]-5.,2) ) > 4 )", degree=1)
chi_medium = dl.interpolate(chi_medium, Vh_OPTIMIZATION)
chi_obstacle = dl.Expression("( ( pow(x[0]-10.,2)+pow(x[1]-5.,2) ) <= 4 )", degree=1)
chi_obstacle = dl.interpolate(chi_obstacle, Vh_OPTIMIZATION)
chi_observation = dl.Expression("( ( pow(x[0]-10.,2)+pow(x[1]-5.,2) ) > 16 )", degree=1)
chi_observation = dl.interpolate(chi_observation, Vh_OPTIMIZATION)
chi_design = dl.Expression("((pow(x[0]-10.,2)+pow(x[1]-5.,2))>=4)*((pow(x[0]-10.,2)+pow(x[1]-5.,2))<=16)", degree=1)
chi_design = dl.interpolate(chi_design, Vh_OPTIMIZATION)

frequency = 300.

c_air = 343.4        # m/s
omega = 2.*math.pi*frequency
c_obstacle = 4600      # m/s # 1531. for sea water


def wavenumber(m, z, obs=False):
    if obs:
        k = dl.Constant(omega/(c_air))
    else:
        k = (dl.Constant(omega/(c_air))*chi_medium + dl.Constant(omega/(c_obstacle))*chi_obstacle) \
            *dl.exp(m*chi_design)*dl.exp(z*chi_design) #*dl.exp((m+dl.Constant(1.0))*z*chi_design) #(1.0+z*z*z*chi) #
        # chi = dl.Expression("((pow(x[0]-10.,2)+pow(x[1]-5.,2))<=16)")
        # k = dl.Constant(omega/(c_air*rho_air))*dl.exp(m*chi)*dl.exp(z*chi)

    return k

# m_fun = dl.Function(Vh_PARAMETER)
# z_fun = dl.Function(Vh_OPTIMIZATION)
# kobs = wavenumber(m_fun, z_fun, obs=True)
# kobs = dl.project(kobs, Vh_OPTIMIZATION)
# kobs = vector2Function(kobs.vector(), Vh_OPTIMIZATION, name="wavenumber")
# kinit = wavenumber(m_fun, z_fun, obs=False)
# kinit = dl.project(kinit, Vh_OPTIMIZATION)
# kinit = vector2Function(kinit.vector(), Vh_OPTIMIZATION, name="wavenumber")
#
# if dlversion() <= (1,6,0):
#     dl.File(comm, "data/k_init.xdmf") << kinit
#     dl.File(comm, "data/k_obs.xdmf") << kobs
# else:
#     xf = dl.XDMFFile(comm, "data/k_init.xdmf")
#     xf.write(kinit)
#     xf = dl.XDMFFile(comm, "data/k_obs.xdmf")
#     xf.write(kobs)

Vhsub = dl.FunctionSpace(mesh, 'CG', 2)
k0 = dl.Constant(omega/(c_air))
ur_incident = dl.Expression('cos(k0*x[0])', k0=k0, degree=2, domain=mesh)
ur_incident = dl.interpolate(ur_incident, Vhsub)
ui_incident = dl.Expression('-sin(k0*x[0])', k0=k0, degree=2, domain=mesh)
ui_incident = dl.interpolate(ui_incident, Vhsub)

sigma_x = dl.Expression("(x[0]<xL)*A*(x[0]-xL)*(x[0]-xL)/(t*t) + (x[0]>xR)*A*(x[0]-xR)*(x[0]-xR)/(t*t)",
                        xL=0., xR=Lx, A=50, t=thickness, degree=2)
sigma_x = dl.interpolate(sigma_x, Vh_OPTIMIZATION)

sigma_y = dl.Expression("(x[1]<yB)*A*(x[1]-yB)*(x[1]-yB)/(t*t) + (x[1]>yT)*A*(x[1]-yT)*(x[1]-yT)/(t*t)",
                        yB=0., yT=Ly, A=50, t=thickness, degree=2)
sigma_y = dl.interpolate(sigma_y, Vh_OPTIMIZATION)


def residual(u, m, p, z, obs=False):
    # reference: E. Turkel, A. Yefet, Absorbing PML boundary layers for wave-like equations, 1998, Applied Numerical Mathematics

    u_trial, v_trial = dl.split(u)
    u_test, v_test = dl.split(p)

    k = wavenumber(m, z, obs)

    ksquared = k**2

    Kr = ksquared - sigma_x*sigma_y
    Ki = -k*(sigma_x + sigma_y)

    Dr_xx = (ksquared+sigma_x*sigma_y)/(ksquared + sigma_x*sigma_x)
    Dr_yy = (ksquared+sigma_x*sigma_y)/(ksquared + sigma_y*sigma_y)
    Di_xx = k*(sigma_x - sigma_y)/(ksquared + sigma_x*sigma_x)
    Di_yy = k*(sigma_y - sigma_x)/(ksquared + sigma_y*sigma_y)

    Dr = dl.as_matrix([[Dr_xx, dl.Constant(0.)], [dl.Constant(0.), Dr_yy]])
    Di = dl.as_matrix([[Di_xx, dl.Constant(0.)], [dl.Constant(0.), Di_yy]])

    ar = dl.inner(Dr*dl.grad(u_trial), dl.grad(u_test))*dl.dx \
       + dl.inner(Di*dl.grad(v_trial), dl.grad(u_test))*dl.dx \
       - Kr*u_trial*u_test*dl.dx \
       - Ki*v_trial*u_test*dl.dx

    ai = -dl.inner(Dr*dl.grad(v_trial), dl.grad(v_test))*dl.dx \
       + dl.inner(Di*dl.grad(u_trial), dl.grad(v_test))*dl.dx \
       + Kr*v_trial*v_test*dl.dx \
       - Ki*u_trial*v_test*dl.dx

    # rhs = dl.Constant(0.0)*u_test*dl.dx + dl.Constant(0.0)*v_test*dl.dx

    # res_form = ar + ai + rhs

    res_form = ar + ai + (ksquared-k0**2)*(ur_incident*u_test+ui_incident*v_test)*dl.dx

    return res_form

bcs = []
bcs0 = []
pde = ControlPDEProblem(Vh, residual, bcs, bcs0, is_fwd_linear=True)

################## 3. Define the quantity of interest (QoI) ############
u_obs = dl.Function(Vh_STATE,name="state")
m = dl.Function(Vh_PARAMETER)
p = dl.TestFunction(Vh_STATE)
z = dl.Function(Vh_OPTIMIZATION)
res_form = residual(u_obs,m,p,z,obs=True)
dl.solve(res_form == 0, u_obs, bcs=bcs)

ur_obs, ui_obs = u_obs.split()

u_total = dl.Function(Vh_STATE, name="state")
ur_total, ui_total = u_total.split()
ur_total, ui_total = dl.interpolate(ur_total, Vhsub), dl.interpolate(ui_total, Vhsub)
ur_obs, ui_obs = dl.interpolate(ur_obs, Vhsub), dl.interpolate(ui_obs, Vhsub)
ur_total.vector().axpy(1.0, ur_obs.vector())
ur_total.vector().axpy(1.0, ur_incident.vector())
ui_total.vector().axpy(1.0, ui_obs.vector())
ui_total.vector().axpy(1.0, ui_incident.vector())

if plotFlag:
    dl.plot(ur_obs, title='observed state real')
    dl.plot(ui_obs, title='observed state imag')

if dlversion() <= (1,6,0):
    dl.File(comm, "data/ur_obs.xdmf") << ur_obs
    dl.File(comm, "data/ui_obs.xdmf") << ui_obs
    dl.File(comm, "data/ur_total.xdmf") << ur_total
    dl.File(comm, "data/ui_total.xdmf") << ur_total
else:
    xf = dl.XDMFFile(comm, "data/ur_obs.xdmf")
    xf.write(ur_obs)
    xf = dl.XDMFFile(comm, "data/ui_obs.xdmf")
    xf.write(ui_obs)
    xf = dl.XDMFFile(comm, "data/ur_total.xdmf")
    xf.write(ur_total)
    xf = dl.XDMFFile(comm, "data/ui_total.xdmf")
    xf.write(ui_total)

output_file = dl.HDF5File(comm, "data/u_obs_load.h5", "w")
output_file.write(u_obs, "u_obs")
output_file.close()

qoi = QoIHelmholtz(mesh, Vh_STATE, chi_observation, u_obs)

################## 4. Define Penalization term ############################

penalization = L1Penalization(Vh[OPTIMIZATION], dl.dx, parameter["alpha"], region=chi_design)

################## 5. Define the prior ##################### ##############
prior = BiLaplacianPrior(Vh[PARAMETER], parameter["gamma"], parameter["delta"])
