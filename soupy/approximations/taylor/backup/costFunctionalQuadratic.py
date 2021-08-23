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

class CostFunctionalQuadratic(CostFunctional):
    """
    The objective is E[Q] + beta Var[Q], by Taylor 2nd order approximation

    The penalization term is P(z) = alpha*||z||_2^2 + ...

    The cost functional is objective + regularization
    """
    def __init__(self, parameter, Vh, pde, qoi, prior, penalization, tol=1e-9):

        self.parameter = parameter
        self.Vh = Vh
        self.pde = pde
        self.generate_optimization = pde.generate_optimization
        self.qoi = qoi
        self.prior = prior
        self.penalization = penalization
        self.z = pde.generate_optimization()
        self.m = prior.mean
        self.tol = tol

        self.x = pde.generate_state()
        self.y = pde.generate_state()
        self.x_all = [self.x, self.m, self.y, self.z]

        self.z_dir = pde.generate_optimization()
        self.z_diff = pde.generate_optimization()
        self.dz = None

        self.rhs_fwd = pde.generate_state()
        self.rhs_adj = pde.generate_state()
        self.rhs_adj2 = pde.generate_state()
        self.rhs_adj3 = pde.generate_state()

        self.xstar = pde.generate_state()
        self.ystar = pde.generate_state()

        self.mhelp = pde.generate_parameter()
        self.Hmhat1 = pde.generate_parameter()
        self.Cdmq = pde.generate_parameter()

        self.func_ncalls = 0
        self.grad_ncalls = 0
        self.hess_ncalls = 0
        self.ncalls = 0

        self.correction = parameter["correction"]
        self.N_mc = parameter["N_mc"]
        self.N_tr = parameter["N_tr"]
        self.beta = parameter["beta"]

        self.xhat = dict()
        self.yhat = dict()
        self.xhatstar = dict()
        self.yhatstar = dict()
        self.mhat = dict()
        self.mhatstar = dict()

        for i in range(self.N_tr):
            self.xhat[i] = pde.generate_state()
            self.yhat[i] = pde.generate_state()
            self.mhat[i] = pde.generate_parameter()
            self.xhatstar[i] = pde.generate_state()
            self.yhatstar[i] = pde.generate_state()
            self.mhatstar[i] = pde.generate_parameter()

        self.d = None
        self.U = None
        self.H = None

        self.lin_mean = 0. # E[Q_lin]
        self.lin_var = 0. # E[(Q_lin-Q_0)^2]
        self.lin_diff_mean = np.zeros(self.N_mc)
        self.lin_fval_mean = np.zeros(self.N_mc)
        self.lin_diff_var = np.zeros(self.N_mc)
        self.lin_fval_var = np.zeros(self.N_mc)

        self.quad_mean = 0. # E[Q_quad]
        self.quad_var = 0. # E[(Q_quad-Q_0)^2]
        self.quad_diff_mean = np.zeros(self.N_mc) # MC[Q-Q_quad]
        self.quad_fval_mean = np.zeros(self.N_mc) # MC[Q] same as self.Q_mc
        self.quad_diff_var = np.zeros(self.N_mc)  # MC[(Q-Q_0)^2 - (Q_quad-Q_0)^2]
        self.quad_fval_var = np.zeros(self.N_mc)  # MC[(Q-Q_0)^2]

        self.const = np.zeros(self.N_mc)

        self.dmr = np.zeros(self.N_mc)
        self.dmmr = np.zeros(self.N_mc)
        self.Q_0 = 0.
        self.Q_mc = np.zeros(self.N_mc)
        self.Cdmq = pde.generate_parameter()

        self.dlcomm = self.Vh[STATE].mesh().mpi_comm()

        self.mpi_rank = dl.MPI.rank(Vh[OPTIMIZATION].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(Vh[OPTIMIZATION].mesh().mpi_comm())
        if self.mpi_size > 1:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD

        # if self.correction:
        #     Omega = pickle.load(open("data/quadratic/Omega.p","rb"))
        # else:
        #     Omega = np.random.randn(self.m.get_local().shape[0], self.N_tr + 10)
        #     pickle.dump( Omega, open( "data/quadratic/Omega.p", "wb" ) )
        # self.Omega = Omega

        randomGen = Random(myid=0, nproc=self.mpi_size)
        if False:   # self.correction
            OmegaSave = pickle.load(open("data/quadratic/Omega.p", "rb"))
            m = pde.generate_parameter()
            Omega = MultiVector(m, self.N_tr+5)
            for i in xrange(self.N_tr+5):
                Omega[i][:] = OmegaSave[i][:]
        else:
            OmegaSave = []
            m = pde.generate_parameter()
            Omega = MultiVector(m, self.N_tr+5)
            for i in xrange(self.N_tr+5):
                randomGen.normal(1., Omega[i])
                OmegaSave.append(Omega[i].get_local())
            # Omega = np.random.randn(sample_mean.get_local().shape[0], N_tr + 10)
            # pickle.dump( OmegaSave, open( "data/quadratic/Omega.p", "wb" ) )

        self.Omega = Omega

        if self.correction:
            self.m_mc = dict()
            self.x_mc = dict()
            self.y_mc = dict()
            for i in range(self.N_mc):
                noise = dl.Vector()
                sample = dl.Vector()
                prior.init_vector(noise, "noise")
                prior.init_vector(sample, 1)

                # noise_size = noise.get_local().shape[0]
                # noise.set_local(np.random.normal(0, 1, noise_size))
                randomGen.normal(1., noise)

                prior.sample(noise, sample, add_mean=False)
                self.m_mc[i] = sample

                self.x_mc[i] = pde.generate_state()
                self.y_mc[i] = pde.generate_state()

        self.tobj = 0.
        self.tgrad = 0.
        self.trand = 0.

        if parameter["dim"] == 1:
            z_trial = dl.TrialFunction(self.Vh[OPTIMIZATION])
            z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
            self.Hessian = dl.assemble(z_trial*z_test*dl.dx)
        elif parameter["dim"] == 2:
            z_trial = dl.TrialFunction(self.Vh[OPTIMIZATION])
            z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
            z1, z2 = dl.split(z_trial)
            w1, w2 = dl.split(z_test)
            self.Hessian = dl.assemble((z1*w1+z2*w2)*dl.dx)
        elif parameter["dim"] == 3:
            z_trial = dl.TrialFunction(self.Vh[OPTIMIZATION])
            z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
            z1, z2, z3 = dl.split(z_trial)
            w1, w2, w3 = dl.split(z_test)
            self.Hessian = dl.assemble((z1*w1+z2*w2+z3*w3)*dl.dx)
        elif parameter["dim"] == 4:
            z_trial = dl.TrialFunction(self.Vh[OPTIMIZATION])
            z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
            z1, z2, z3, z4 = dl.split(z_trial)
            w1, w2, w3, w4 = dl.split(z_test)
            self.Hessian = dl.assemble((z1*w1+z2*w2+z3*w3+z4*w4)*dl.dx)
        else:
            z_vec = [None]*parameter["dim"]
            c_vec = [None]*parameter["dim"]
            for i in range(parameter["dim"]):
                z_trial = dl.TrialFunction(self.Vh[OPTIMIZATION].sub(0))
                z_test = dl.TestFunction(self.Vh[OPTIMIZATION].sub(0))
                z_vec[i] = z_trial
                c_vec[i] = z_test
            z_vec = dl.as_vector(z_vec)
            c_vec = dl.as_vector(c_vec)
            self.Hessian = dl.assemble(dl.dot(c_vec, z_vec)*dl.dx)

        if dlversion() <= (1,6,0):
            self.Msolver = dl.PETScKrylovSolver("cg", "jacobi")
        else:
            self.Msolver = dl.PETScKrylovSolver(self.Vh[OPTIMIZATION].mesh().mpi_comm(), "cg", "jacobi")
        self.Msolver.set_operator(self.Hessian)
        self.Msolver.parameters["maximum_iterations"] = 100
        self.Msolver.parameters["relative_tolerance"] = 1.e-12
        self.Msolver.parameters["error_on_nonconvergence"] = True
        self.Msolver.parameters["nonzero_initial_guess"] = False

    def objectiveLinear(self):
        # solve the forward problem at the mean
        self.x_all[OPTIMIZATION] = self.z
        self.pde.solveFwd(self.x, self.x_all, self.tol)
        Q_0 = self.qoi.eval(self.x) # Q(x(m_0))
        self.Q_0 = Q_0
        self.x_all[STATE] = self.x

        # solve the adjoint problem at the mean
        rhs = self.pde.generate_state()
        self.qoi.adj_rhs(self.x, rhs)
        self.pde.solveAdj(self.y, self.x_all, rhs, self.tol)
        self.x_all[ADJOINT] = self.y

        # set the linearization point for incremental forward and adjoint problems
        self.pde.setLinearizationPoint(self.x_all)
        self.qoi.setLinearizationPoint(self.x)

        # E[Q_lin]
        self.lin_mean = Q_0
        # beta*Var[Q_lin]
        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])

        f_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)
        m_test = dl.TestFunction(self.Vh[PARAMETER])
        dmq = dl.assemble(dl.derivative(f_form,m_fun,m_test))
        Cdmq = self.pde.generate_parameter()
        self.prior.Rsolver.solve(Cdmq, dmq)
        self.Cdmq = Cdmq
        self.lin_var = Q_0**2 + dmq.inner(Cdmq)

        # Monte Carlo corrections
        if self.correction:
            for i in range(self.N_mc):
                # solve the forward problem at the new sample
                m = self.m_mc[i]
                m_i = self.pde.generate_parameter()
                m_i.axpy(1.,m)
                m_i.axpy(1.,self.m)
                x_all = [self.x, m_i, self.y, self.z]
                x = self.pde.generate_state()
                self.pde.solveFwd(x, x_all, self.tol)
                self.x_mc[i] = x
                Q_i = self.qoi.eval(x) #Q(x(m_i))
                self.Q_mc[i] = Q_i

                f_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)
                m_hat = vector2Function(self.m_mc[i], self.Vh[PARAMETER])
                dmr = dl.assemble(dl.derivative(f_form,m_fun,m_hat))
                self.dmr[i] = dmr

                # Monte Carlo correction E[Q-Q_lin]
                self.lin_diff_mean[i] = Q_i - (Q_0+dmr)
                self.lin_fval_mean[i] = Q_i
                # Monte Carlo correction E[(Q)^2 - (Q_lin)^2]
                self.lin_diff_var[i] = (Q_i)**2 - (Q_0+dmr)**2
                self.lin_fval_var[i] = (Q_i)**2

        cost = self.lin_mean + np.mean(self.lin_diff_mean) \
                     + self.beta*(self.lin_var + np.mean(self.lin_diff_var)) \
                     - self.beta*(self.lin_mean+np.mean(self.lin_diff_mean))**2

        if self.mpi_rank == 0:
            print("#######################linear approximation#########################")
            header = ["mean", "mean_diff", "var", "var_diff"]
            print('{:<20} {:<20} {:<20} {:<20}'.format(*header))
            data = [self.lin_mean, np.mean(self.lin_diff_mean), self.lin_var - self.lin_mean**2, np.mean(self.lin_diff_var)]
            print('{:<20} {:<20} {:<20} {:<20}'.format(*data))
            print("####################################################################")

            if self.correction:
                print ("########################################################################")
                header = ["mean_mse", "lin_mean_diff_mse", "var_mse", "lin_var_diff_mse"]
                print('{:<20} {:<20} {:<20} {:<20}'.format(*header))
                data = [np.var(self.lin_fval_mean)/self.N_mc, np.var(self.lin_diff_mean)/self.N_mc,
                        np.var(self.lin_fval_var)/self.N_mc, np.var(self.lin_diff_var)/self.N_mc]
                print('{:<20} {:<20} {:<20} {:<20}'.format(*data))
                print ("#########################################################################")

        return cost

    def objective(self):

        self.objectiveLinear()


        trand = time.time()

        self.H = ReducedHessianSVD(self.pde, self.qoi, self.tol)
        self.d, self.U = doublePassG(self.H, self.prior.R, self.prior.Rsolver, self.Omega, self.N_tr, s = 1)
                    # check_Bortho = True, check_Aortho=True, check_residual = True)

        for i in range(self.N_tr):
            self.mhat[i].zero()
            self.mhat[i].axpy(1., self.U[i])

        # np.savez("data/ds1.npz",d=self.d)

        self.trand = time.time()-trand

        if self.mpi_rank == 0:
            print ("#####################################################################")
            print (time.time()-trand, "seconds for randomized eigensolver")
            print ("#####################################################################")


        self.quad_mean = self.Q_0 + 0.5*np.sum(self.d)
        self.quad_var = self.lin_var + 0.25*(np.sum(self.d))**2 + 0.5*np.sum(np.square(self.d)) + self.Q_0*np.sum(self.d)

        for i in range(self.N_tr):
            dmmr, xhat, yhat = self.HessianInner(self.mhat[i], self.mhat[i])
            self.xhat[i] = xhat
            self.yhat[i] = yhat

        if self.correction:
            for i in range(self.N_mc):
                dmmr, xhat, yhat = self.HessianInner(self.m_mc[i], self.m_mc[i])
                self.dmmr[i] = dmmr
                self.xhat[i+self.N_tr] = xhat
                self.yhat[i+self.N_tr] = yhat

                self.quad_diff_mean[i] = self.Q_mc[i] - (self.Q_0+self.dmr[i]+1./2*self.dmmr[i])
                self.quad_fval_mean[i] = self.Q_mc[i]

                self.quad_diff_var[i] = (self.Q_mc[i])**2 - (self.Q_0+self.dmr[i]+1./2*self.dmmr[i])**2
                self.quad_fval_var[i] = (self.Q_mc[i])**2 # (Q_i)^2

                self.const[i] = -0.5/self.N_mc \
                                -self.beta/self.N_mc*(self.Q_0+self.dmr[i]+1./2*self.dmmr[i]) \
                                +self.beta/self.N_mc*(self.quad_mean + np.mean(self.quad_diff_mean))
                # print "######################################################################################"
                # header = ["ith sample",  "lin_diff_mean", "lin_diff_var", "quad_diff_mean", "quad_diff_var"]
                # print('{:<20} {:<20} {:<20} {:<20} {:<20}'.format(*header))
                # data = [i+1, self.lin_diff_mean[i]/self.lin_fval_mean[i], self.lin_diff_var[i]/self.lin_fval_var[i],
                #            self.quad_diff_mean[i]/self.quad_fval_mean[i], self.quad_diff_var[i]/self.quad_fval_var[i]]
                # print('{:<20} {:<20} {:<20} {:<20} {:<20}'.format(*data))
                # print "######################################################################################"

        for i in range(self.N_tr):
            self.mhatstar[i].zero()
            if self.correction:
                self.mhatstar[i].axpy(0.5+self.beta*self.d[i]-self.beta*np.mean(self.quad_diff_mean), self.U[i])
            else:
                self.mhatstar[i].axpy(0.5+self.beta*self.d[i], self.U[i])

        cost = self.quad_mean + np.mean(self.quad_diff_mean) \
                    + self.beta*(self.quad_var + np.mean(self.quad_diff_var)) \
                    - self.beta*(self.quad_mean + np.mean(self.quad_diff_mean))**2

        if self.mpi_rank == 0:
            print("#######################quadratic approximation#########################")
            header = ["mean", "mean_diff", "var", "var_diff"]
            print('{:<20} {:<20} {:<20} {:<20}'.format(*header))
            data = [self.quad_mean, np.mean(self.quad_diff_mean), self.quad_var - self.quad_mean**2, np.mean(self.quad_diff_var)]
            print('{:<20} {:<20} {:<20} {:<20}'.format(*data))
            print("####################################################################")


            if self.correction:
                print ("######################################################################################")
                header = ["mean_mse", "quad_mean_diff_mse", "var_mse", "quad_var_diff_mse"]
                print('{:<20} {:<20} {:<20} {:<20}'.format(*header))
                data = [np.var(self.quad_fval_mean)/self.N_mc, np.var(self.quad_diff_mean)/self.N_mc,
                        np.var(self.quad_fval_var)/self.N_mc, np.var(self.quad_diff_var)/self.N_mc]
                print('{:<20} {:<20} {:<20} {:<20}'.format(*data))
                print ("######################################################################################")

        return cost

    def QuadraticSampling(self, sample, savedata=False, i=0, type='random'):

        m = self.pde.generate_parameter()
        m.axpy(1., self.m)
        m.axpy(1., sample)
        x_all = [self.x, m, self.y, self.z]
        x = self.pde.generate_state()
        self.pde.solveFwd(x, x_all, self.tol)

        x_fun = vector2Function(self.x, self.Vh[STATE])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])

        f_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)
        sample_fun = vector2Function(sample, self.Vh[PARAMETER])
        dmr = dl.assemble(dl.derivative(f_form, m_fun, sample_fun))
        dmmr, xhat, yhat = self.HessianInner(sample, sample)

        Q_mean = self.qoi.eval(x) # Q(x(m_i))
        lin_diff_mean = Q_mean - self.Q_0 - dmr # the gradient term
        quad_diff_mean = lin_diff_mean - dmmr/2

        Q_var = (Q_mean)**2
        lin_diff_var = Q_var - (self.Q_0+dmr)**2
        quad_diff_var = Q_var - (self.Q_0+dmr+dmmr/2)**2

        if savedata:
            x_fun = vector2Function(x, self.Vh[STATE])
            filename = "data/u_" + type + "_"+ str(i) +"_load.h5"
            output_file = dl.HDF5File(self.dlcomm, filename, "w")
            output_file.write(x_fun, "state")
            output_file.close()

        return x, Q_mean, lin_diff_mean, quad_diff_mean, Q_var, lin_diff_var, quad_diff_var


    def ForwardSolution(self,sample):

        m = self.pde.generate_parameter()
        m.axpy(1., self.m)
        m.axpy(1., sample)
        x_all = [self.x, m, self.y, self.z]
        x = self.pde.generate_state()
        self.pde.solveFwd(x, x_all, self.tol)

        x_all[STATE] = x

        # solve the adjoint problem at the mean
        y = self.pde.generate_state()
        rhs = self.pde.generate_state()
        self.qoi.adj_rhs(x,rhs)
        self.pde.solveAdj(y,x_all,rhs,self.tol)

        return x, y

    # Hessian Action H*m
    def HessianAction(self, mhat1):
        # check Hessian by finite difference
        xhat = self.pde.generate_state()
        yhat = self.pde.generate_state()
        self.pde.apply_ij(ADJOINT,PARAMETER, mhat1, self.rhs_fwd)
        self.pde.solveIncremental(xhat, -self.rhs_fwd, False, self.tol) # False for forward,
        self.pde.apply_ij(STATE, STATE, xhat, self.rhs_adj)
        self.pde.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj2)
        self.rhs_adj.axpy(1., self.rhs_adj2)
        self.qoi.apply_ij(STATE,STATE,xhat,self.rhs_adj3)
        self.rhs_adj.axpy(1., self.rhs_adj3)
        self.pde.solveIncremental(yhat, -self.rhs_adj, True, self.tol) # True for adjoint,
        self.pde.apply_ij(PARAMETER,PARAMETER, mhat1, self.Hmhat1)
        self.pde.apply_ij(PARAMETER,ADJOINT,yhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)
        self.pde.apply_ij(PARAMETER,STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)

        return self.Hmhat1

    # inner product (mhat1, mhat2)_Hessian
    def HessianInner(self, mhat1, mhat2):
        # check Hessian by finite difference
        xhat = self.pde.generate_state()
        yhat = self.pde.generate_state()
        self.pde.apply_ij(ADJOINT,PARAMETER, mhat1, self.rhs_fwd)
        self.pde.solveIncremental(xhat, -self.rhs_fwd, False, self.tol) # False for forward,
        self.pde.apply_ij(STATE, STATE, xhat, self.rhs_adj)
        self.pde.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj2)
        self.rhs_adj.axpy(1., self.rhs_adj2)
        self.qoi.apply_ij(STATE,STATE,xhat,self.rhs_adj3)
        self.rhs_adj.axpy(1., self.rhs_adj3)
        self.pde.solveIncremental(yhat, -self.rhs_adj, True, self.tol) # True for adjoint,
        self.pde.apply_ij(PARAMETER,PARAMETER, mhat1, self.Hmhat1)
        self.pde.apply_ij(PARAMETER,ADJOINT,yhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)
        self.pde.apply_ij(PARAMETER,STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)

        return mhat2.inner(self.Hmhat1), xhat, yhat

    def costValue(self, z):
        self.func_ncalls += 1

        if isinstance(z,np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0]:idx[1]])
        else:
            self.z.zero()
            self.z.axpy(1.0, z)

        tobj = time.time()

        objective = self.objective()

        self.tobj = time.time() - tobj

        if self.mpi_rank == 0:
            print ("#####################################################################")
            print (time.time()-tobj, "seconds for objective evaluation")
            print ("#####################################################################")

        penalization = self.penalization.eval(self.z)

        cost = objective + penalization

        if self.mpi_rank == 0:
            header = ["# func_calls", "Cost", "Objective", "Penalization"]
            print("\n {:<20} {:<20} {:<20} {:<20}".format(*header))
            data = [self.func_ncalls, cost, objective, penalization]
            print('{:<20d} {:<20.2e} {:<20.2e} {:<20.2e}'.format(*data))

        return cost

    def solveAdjIncrementalAdj(self):

        for i in range(self.N_tr):
            xhatstar = self.pde.generate_state()
            if self.correction:
                xhatstar.axpy((0.5+self.beta*self.d[i]-self.beta*np.mean(self.quad_diff_mean)), self.xhat[i])
                self.xhatstar[i] = xhatstar
            else:
                xhatstar.axpy((0.5+self.beta*self.d[i]), self.xhat[i])
                self.xhatstar[i] = xhatstar

        if self.correction:
            for i in range(self.N_mc):
                xhatstar = self.pde.generate_state()
                dmyr = self.pde.forSolveAdjIncrementalAdj(self.x_all, self.m_mc[i])
                xhatstarrhs = self.pde.generate_state()

                xhatstarrhs.axpy(self.const[i], dmyr)
                self.pde.solver_fwd_inc.solve(xhatstar, -xhatstarrhs)
                self.xhatstar[i+self.N_tr] = xhatstar

    def solveAdjIncrementalFwd(self):

        for i in range(self.N_tr):
            yhatstar = self.pde.generate_state()
            if self.correction:
                yhatstar.axpy((0.5+self.beta*self.d[i]-self.beta*np.mean(self.quad_diff_mean)), self.yhat[i])
                self.yhatstar[i] = yhatstar
            else:
                yhatstar.axpy((0.5+self.beta*self.d[i]), self.yhat[i])
                self.yhatstar[i] = yhatstar

        if self.correction:

            for i in range(self.N_mc):
                yhatstar = self.pde.generate_state()
                dmxr, dxxr, dxxq = \
                    self.pde.forSolveAdjIncrementalFwd(self.x_all, self.m_mc[i], self.xhatstar[i+self.N_tr], self.qoi)
                yhatstarrhs = self.pde.generate_state()

                yhatstarrhs.axpy(self.const[i], dmxr)
                yhatstarrhs.axpy(1., dxxr)
                yhatstarrhs.axpy(1., dxxq)
                self.pde.solver_adj_inc.solve(yhatstar, -yhatstarrhs)
                self.yhatstar[i+self.N_tr] = yhatstar

    def solveAdjAdj(self):

        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])

        f_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)
        y_test = dl.TestFunction(self.Vh[ADJOINT])
        xstarrhs = self.pde.generate_state()

        Cdmq_fun = vector2Function(self.Cdmq, self.Vh[PARAMETER])
        dmr = dl.derivative(f_form, m_fun, Cdmq_fun)
        dmyr = dl.derivative(dmr, y_fun, y_test)
        dmyr = dl.assemble(dmyr)
        [bc.apply(dmyr) for bc in self.pde.bc0]
        # self.pde.bc0.apply(dmyr)
        xstarrhs.axpy(2*self.beta, dmyr)

        for i in range(self.N_tr):
            dmyr, dxxyr, dxmyr, dmmyr, dmxyr = \
                self.pde.forSolveAdjAdj(self.x_all,self.xhat[i],self.xhatstar[i],self.mhat[i],self.mhatstar[i])

            xstarrhs.axpy(1., dxxyr)
            xstarrhs.axpy(1., dxmyr)

            xstarrhs.axpy(1., dmmyr)
            xstarrhs.axpy(1., dmxyr)

        if self.correction:
            for i in range(self.N_mc):
                dmyr, dxxyr, dxmyr, dmmyr, dmxyr = \
                    self.pde.forSolveAdjAdj(self.x_all,self.xhat[i+self.N_tr],self.xhatstar[i+self.N_tr],self.m_mc[i],self.m_mc[i])

                xstarrhs.axpy(1., dxxyr)
                xstarrhs.axpy(1., dxmyr)

                xstarrhs.axpy(2.*self.const[i], dmyr)
                xstarrhs.axpy(self.const[i], dmmyr)
                xstarrhs.axpy(self.const[i], dmxyr)

        self.pde.solver_fwd_inc.solve(self.xstar, -xstarrhs)

    def solveAdjFwd(self):
        ystarrhs = self.pde.generate_state()
        ystarrhsqoi = self.pde.generate_state()
        self.qoi.grad_state(self.x, ystarrhsqoi)
        ystarrhs.axpy(1., ystarrhsqoi)
        ystarrhspde = self.pde.generate_state()
        self.qoi.apply_ij(STATE, STATE, self.xstar, ystarrhspde)
        ystarrhs.axpy(1., ystarrhspde)
        self.pde.apply_ij(STATE, STATE, self.xstar, ystarrhspde)
        ystarrhs.axpy(1., ystarrhspde)
        [bc.apply(ystarrhs) for bc in self.pde.bc0]
        # self.pde.bc0.apply(ystarrhs)

        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])

        f_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)
        x_test = dl.TestFunction(self.Vh[STATE])

        Cdmq_fun = vector2Function(self.Cdmq, self.Vh[PARAMETER])
        dmr = dl.derivative(f_form, m_fun, Cdmq_fun)
        dmxr = dl.derivative(dmr, x_fun, x_test)
        dmxr = dl.assemble(dmxr)
        [bc.apply(dmxr) for bc in self.pde.bc0]
        # self.pde.bc0.apply(dmxr)
        ystarrhs.axpy(2.*self.beta, dmxr)

        for i in range(self.N_tr):
            mhat = self.mhat[i]
            mhatstar = self.mhatstar[i]
            xhat = self.xhat[i]
            xhatstar = self.xhatstar[i]
            yhat = self.yhat[i]
            yhatstar = self.yhatstar[i]

            dmxr, dyxxr, dymxr, dxyxr, dxxxr, dxmxr, dxxxq, dmmxr, dmyxr, dmxxr = \
                self.pde.forSolveAdjFwd(self.x_all, xhat, xhatstar, mhat, mhatstar, yhat, yhatstar, self.qoi)

            ystarrhs.axpy(1., dyxxr)
            ystarrhs.axpy(1., dymxr)

            ystarrhs.axpy(1., dxyxr)
            ystarrhs.axpy(1., dxxxr)
            ystarrhs.axpy(1., dxmxr)
            ystarrhs.axpy(1., dxxxq)

            ystarrhs.axpy(1., dmmxr)
            ystarrhs.axpy(1., dmxxr)
            ystarrhs.axpy(1., dmyxr)

        if self.correction:
            ystarrhs.axpy(-2.*self.beta*np.mean(self.quad_diff_mean),
                          ystarrhsqoi)
            for i in range(self.N_mc):
                mhat = self.m_mc[i]
                mhatstar = self.m_mc[i]
                xhat = self.xhat[i+self.N_tr]
                xhatstar = self.xhatstar[i+self.N_tr]
                yhat = self.yhat[i+self.N_tr]
                yhatstar = self.yhatstar[i+self.N_tr]

                dmxr, dyxxr, dymxr, dxyxr, dxxxr, dxmxr, dxxxq, dmmxr, dmyxr, dmxxr = \
                    self.pde.forSolveAdjFwd(self.x_all,xhat,xhatstar,mhat,mhatstar,yhat,yhatstar,self.qoi)

                ystarrhs.axpy(1., dyxxr)
                ystarrhs.axpy(1., dymxr)

                ystarrhs.axpy(1., dxyxr)
                ystarrhs.axpy(1., dxxxr)
                ystarrhs.axpy(1., dxmxr)
                ystarrhs.axpy(1., dxxxq)

                ystarrhs.axpy(2.*self.const[i], dmxr)

                ystarrhs.axpy(2*self.const[i], ystarrhsqoi)

                ystarrhs.axpy(self.const[i], dmmxr)
                ystarrhs.axpy(self.const[i], dmxxr)
                ystarrhs.axpy(self.const[i], dmyxr)

        self.pde.solver_adj_inc.solve(self.ystar, -ystarrhs)

    def solve_ymc(self):
        for i in range(self.N_mc):
            m = self.m_mc[i]
            m_i = self.pde.generate_parameter()
            m_i.axpy(1., m)
            m_i.axpy(1., self.m)
            x_all = [self.x, m_i, self.y, self.z]
            rhs = self.pde.generate_state()
            self.qoi.adj_rhs(self.x_mc[i], rhs)
            [bc.apply(rhs) for bc in self.pde.bc0]

            rhs[:] *= 1./self.N_mc*(1.
                                                    + 2.*self.beta*(self.quad_fval_mean[i])
                                                    - 2.*self.beta*(self.quad_mean+np.mean(self.quad_diff_mean)))

            y_mc = self.pde.generate_state()
            self.pde.solveAdj(y_mc, x_all, rhs, self.tol)
            self.y_mc[i] = y_mc

    def costGradient(self, z):
        self.grad_ncalls += 1
        self.ncalls = 0

        if isinstance(z,np.ndarray):
            zprev = self.z.gather_on_zero()
            if self.mpi_size > 1:
                zprev = self.comm.bcast(zprev, root=0)

            if np.linalg.norm(z-zprev) > 1.e-15:
                self.costValue(z)

        else:
            self.z_diff.zero()
            self.z_diff.axpy(1.0, self.z)
            self.z_diff.axpy(-1.0, z)
            if self.z_diff.inner(self.z_diff) > 1.e-20:
                self.costValue(z)

        tgrad = time.time()

        self.solveAdjIncrementalAdj()
        self.solveAdjIncrementalFwd()
        self.solveAdjAdj()
        self.solveAdjFwd()
        if self.correction:
            self.solve_ymc()

        if self.mpi_rank == 0:
            print ("#####################################################################")
            print (time.time() - tgrad, "seconds for solving adjoint problems")
            print ("#####################################################################")

        dzq = self.pde.generate_optimization()
        x_fun = vector2Function(self.x, self.Vh[STATE])
        y_fun = vector2Function(self.y, self.Vh[ADJOINT])
        m_fun = vector2Function(self.m, self.Vh[PARAMETER])
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])
        Cdmq_fun = vector2Function(self.Cdmq, self.Vh[PARAMETER])

        f_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)
        z_test = dl.TestFunction(self.Vh[OPTIMIZATION])

        dmr = dl.derivative(f_form, m_fun, Cdmq_fun)
        dmzr = dl.derivative(dmr, z_fun, z_test)
        dmzr = dl.assemble(dmzr)
        dzq.axpy(2.*self.beta, dmzr)

        ystar_fun = vector2Function(self.ystar, self.Vh[ADJOINT])
        dyr = dl.derivative(f_form, y_fun, ystar_fun)
        dyzr = dl.derivative(dyr, z_fun, z_test)
        dyzr = dl.assemble(dyzr)

        dzq.axpy(1., dyzr)

        xstar_fun = vector2Function(self.xstar, self.Vh[STATE])
        dxr = dl.derivative(f_form, x_fun, xstar_fun)
        dxzr = dl.derivative(dxr, z_fun, z_test)
        dxzr = dl.assemble(dxzr)

        dzq.axpy(1., dxzr)

        if self.correction:
            for i in range(self.N_mc):
                m = self.m_mc[i]
                m_i = self.pde.generate_parameter()
                m_i.axpy(1.,m)
                m_i.axpy(1.,self.m)
                m_fun = vector2Function(m_i, self.Vh[PARAMETER])
                x_mc = self.x_mc[i]
                x_fun = vector2Function(x_mc, self.Vh[STATE])
                y_mc = self.y_mc[i]
                y_fun = vector2Function(y_mc, self.Vh[ADJOINT])
                f_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)

                dyr = dl.derivative(f_form, y_fun, y_fun)
                dyzr = dl.derivative(dyr, z_fun, z_test)
                dyzr = dl.assemble(dyzr)

                dzq.axpy(1., dyzr)

        for i in range(self.N_tr):
            mhat = self.mhat[i]
            mhatstar = self.mhatstar[i]
            xhat = self.xhat[i]
            xhatstar = self.xhatstar[i]
            yhat = self.yhat[i]
            yhatstar = self.yhatstar[i]

            dyzr, dxzr, dyxzr, dymzr, dxyzr, dxxzr, dxmzr, dmmzr, dmyzr, dmxzr = \
                self.pde.gradientControl(self.x_all,self.xstar,self.ystar,xhat,xhatstar,mhat,mhatstar,yhat,yhatstar)

            dzq.axpy(1.,dyxzr)
            dzq.axpy(1.,dymzr)

            dzq.axpy(1.,dxyzr)
            dzq.axpy(1.,dxxzr)
            dzq.axpy(1.,dxmzr)

            dzq.axpy(1.,dmmzr)
            dzq.axpy(1.,dmxzr)
            dzq.axpy(1.,dmyzr)

        if self.correction:
            for i in range(self.N_mc):
                mhat = self.m_mc[i]
                mhatstar = self.m_mc[i]
                xhat = self.xhat[i+self.N_tr]
                xhatstar = self.xhatstar[i+self.N_tr]
                yhat = self.yhat[i+self.N_tr]
                yhatstar = self.yhatstar[i+self.N_tr]

                dyzr, dxzr, dyxzr, dymzr, dxyzr, dxxzr, dxmzr, dmmzr, dmyzr, dmxzr= \
                    self.pde.gradientControl(self.x_all,self.xstar,self.ystar,xhat,xhatstar,mhat,mhatstar,yhat,yhatstar)

                dzq.axpy(1.,dyxzr)
                dzq.axpy(1.,dymzr)

                dzq.axpy(1.,dxyzr)
                dzq.axpy(1.,dxxzr)
                dzq.axpy(1.,dxmzr)

                dzq.axpy(2*self.const[i], dmzr)
                dzq.axpy(self.const[i], dmmzr)
                dzq.axpy(self.const[i], dmxzr)
                dzq.axpy(self.const[i], dmyzr)


        dz = self.pde.generate_optimization()
        dz.axpy(1., dzq)

        dzp = self.pde.generate_optimization()
        self.penalization.grad(self.z, dzp)
        dz.axpy(1., dzp)

        self.tgrad = time.time() - tgrad

        if self.mpi_rank == 0:
            print ("#####################################################################")
            print (time.time() - tgrad, "seconds for gradient evaluation")
            print ("#####################################################################")


        dzCost = np.sqrt(dz.inner(dz))
        dzObjective = np.sqrt(dzq.inner(dzq))
        dzPenalization = np.sqrt(dzp.inner(dzp))
        if self.mpi_rank == 0:
            header = ["# grad_calls", "gradCost", "gradObjective", "gradPenalization"]
            print("\n {:<20} {:<20} {:<20} {:<20}".format(*header))
            data = [self.grad_ncalls, dzCost, dzObjective, dzPenalization]
            print('{:<20d} {:<20.2e} {:<20.2e} {:<20.2e}'.format(*data))

            # if self.mpi_rank == 0:
            #     print("grad_ncalls = ", self.grad_ncalls, "||gradient||_2 = ", np.linalg.norm(dz))
        if isinstance(z,np.ndarray):
            dz = dz.gather_on_zero()
            if self.mpi_size > 1:
                dz = self.comm.bcast(dz, root=0)

        self.dz = dz

        tmp = self.generate_optimization()
        self.Msolver.solve(tmp, dz)
        dznorm = np.sqrt(dz.inner(tmp))

        return dz, dznorm

    def costHessian(self, z, z_dir, FD = True):
        self.hess_ncalls += 1
        self.ncalls += 1

        if isinstance(z,np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0]:idx[1]])
        else:
            self.z.zero()
            self.z.axpy(1.0, z)
        if isinstance(z_dir,np.ndarray):
            idx = self.z_dir.local_range()
            self.z_dir.set_local(z_dir[idx[0]:idx[1]])
        else:
            self.z_dir.zero()
            self.z_dir.axpy(1.0, z_dir)

        if FD:
            epsilon = 1.e-6
            if isinstance(z,np.ndarray):
                z1 = z - epsilon*z_dir
                z2 = z + epsilon*z_dir
            else:
                z1 = z.axpy(-epsilon, z_dir)
                z2 = z.axpy(epsilon, z_dir)

            self.costValue(z1)
            dz1 = self.costGradient(z1)
            self.costValue(z2)
            dz2 = self.costGradient(z2)

            if isinstance(z,np.ndarray):
                Hz = (dz2-dz1)/(2*epsilon)
            else:
                Hz = dz2.axpy(-1.0, dz1)/(2*epsilon)

            if self.mpi_rank == 0:
                print("hess_ncalls = ", self.hess_ncalls)

            return Hz

        else:
            thess = time.time()

            z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])
            z_dir_fun = vector2Function(self.z_dir, self.Vh[OPTIMIZATION])
            m_fun = vector2Function(self.m, self.Vh[PARAMETER])
            x_fun = vector2Function(self.x, self.Vh[STATE])
            y_fun = vector2Function(self.y, self.Vh[ADJOINT])

            r_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)

            # solve incremental state
            x_trial = dl.TrialFunction(self.Vh[STATE])
            x_star = dl.Function(self.Vh[STATE])
            y_test = dl.TestFunction(self.Vh[ADJOINT])

            rx_form = dl.derivative(r_form, x_fun, x_trial)
            rxy_form = dl.derivative(rx_form, y_fun, y_test)

            rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
            rzy_form = dl.derivative(rz_form, y_fun, y_test)
            Ly_form = -rzy_form

            dl.solve(rxy_form == Ly_form, x_star, self.pde.bc0)  # specify Kylov solver

            # solve incremental adjoint
            x_test = dl.TestFunction(self.Vh[STATE])
            y_trial = dl.TrialFunction(self.Vh[ADJOINT])
            y_star = dl.Function(self.Vh[ADJOINT])

            ry_form = dl.derivative(r_form, y_fun, y_trial)
            ryx_form = dl.derivative(ry_form, x_fun, x_test)

            rx_form = dl.derivative(r_form, x_fun, x_star)
            rxx_form = dl.derivative(rx_form, x_fun, x_test)

            qx_form = dl.derivative(self.qoi.form(x_fun), x_fun, x_star)
            qxx_form = dl.derivative(qx_form, x_fun, x_test)

            rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
            rzx_form = dl.derivative(rz_form, x_fun, x_test)

            Lx_form = -(rxx_form + qxx_form + rzx_form)

            dl.solve(ryx_form == Lx_form, y_star, self.pde.bc0)

            # assemble Hessian
            z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
            ry_form = dl.derivative(r_form, y_fun, y_star)
            ryz_form = dl.derivative(ry_form, z_fun, z_test)

            rx_form = dl.derivative(r_form, x_fun, x_star)
            rxz_form = dl.derivative(rx_form, z_fun, z_test)

            rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
            rzz_form = dl.derivative(rz_form, z_fun, z_test)

            Lz_form = ryz_form + rxz_form + rzz_form

            Hz = dl.assemble(Lz_form)

            dzzp = self.pde.generate_optimization()
            self.penalization.hessian(self.z, self.z_dir, dzzp)
            Hz.axpy(1., dzzp)

            if isinstance(z, np.ndarray):
                Hz = Hz.gather_on_zero()
                if self.mpi_size > 1:
                    Hz = self.comm.bcast(Hz, root=0)

            if self.mpi_rank == 0:
                print("#####################################################################")
                print(time.time() - thess, "seconds for Hessian evaluation", "hess_ncalls = ", self.hess_ncalls,)
                print("#####################################################################")

            return Hz


    # define reduced z-Hessian used for Newton CG algorithm with function Hessian.init_vector and Hessian.mult
    def init_vector(self, z, dim):
        z_trial = dl.TrialFunction(self.Vh[OPTIMIZATION])
        z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
        self.Hessian = dl.assemble(z_trial*z_test*dl.dx)

        self.Hessian.init_vector(z, dim)

        # znew = self.pde.generate_optimization()
        # z.init(znew.size())

    def mult(self, zhat, Hzhat, FD=False):

        z = self.z.copy()

        Hz = self.costHessian(z, zhat, FD=FD)

        Hzhat.zero()
        Hzhat.axpy(1.0, Hz)