from __future__ import absolute_import, division, print_function

import numpy as np
import time
import dolfin as dl
dl.dx = dl.dx(metadata={'quadrature_degree':2, "representation":'uflacs'}) #, "representation":'uflacs'

# import sys
# sys.path.append('../')
from ..costFunctional import CostFunctional
# sys.path.append('../../')
from ...utils.variables import STATE, PARAMETER, ADJOINT, OPTIMIZATION
from ...utils.vector2function import vector2Function
from ...utils.random import Random
from ...utils.checkDolfinVersion import dlversion

class CostFunctionalLinearMultiPDE(CostFunctional):
    """
    The objective is E[Q] + beta Var[Q] = Q(u(m_0)) + beta <Cdm, dm> , by Taylor first order approximation

    The penalization term is P(z) = alpha*||z||_2^2 + ...

    The cost functional is objective + regularization
    """
    def __init__(self, parameter, Vh, pde, qoi, prior, penalization, tol=1e-9):

        self.parameter = parameter
        self.Vh = Vh
        self.pde = pde
        self.N_pde = len(pde)
        self.generate_optimization = self.pde[0].generate_optimization
        self.qoi = qoi
        self.prior = prior
        self.penalization = penalization
        self.m = prior.mean
        self.tol = tol

        self.z = self.generate_optimization()
        self.z_dir = self.generate_optimization()
        self.z_diff = self.generate_optimization()
        self.dz = None

        self.func_ncalls = 0
        self.grad_ncalls = 0
        self.hess_ncalls = 0
        self.ncalls = 0
        self.cg_ncalls = []

        self.correction = parameter["correction"]
        self.N_mc = parameter["N_mc"]
        self.beta = parameter["beta"]
        self.sign = 1.

        self.x, self.y, self.x_all, = [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde
        self.xstar, self.ystar, self.Cdmq = [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde
        self.lin_mean, self.lin_var, self.lin_diff_mean, self.lin_fval_mean, self.lin_diff_var, self.lin_fval_var = \
            [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde
        self.dmr = [None]*self.N_pde
        self.quad_mean, self.quad_var, self.quad_diff_mean, self.quad_fval_mean, self.quad_diff_var, self.quad_fval_var = \
            [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde
        for n_pde in range(self.N_pde):
            self.x[n_pde] = pde[n_pde].generate_state()
            self.y[n_pde] = pde[n_pde].generate_state()
            self.x_all[n_pde] = [self.x[n_pde], self.m, self.y[n_pde], self.z]

            self.xstar[n_pde] = pde[n_pde].generate_state()
            self.ystar[n_pde] = pde[n_pde].generate_state()

            self.Cdmq[n_pde] = pde[n_pde].generate_parameter()

            self.lin_mean[n_pde] = 0. # E[Q_lin]
            self.lin_var[n_pde] = 0. # E[(Q_lin)^2]
            self.lin_diff_mean[n_pde] = np.zeros(self.N_mc)
            self.lin_fval_mean[n_pde] = np.zeros(self.N_mc)
            self.lin_diff_var[n_pde] = np.zeros(self.N_mc)
            self.lin_fval_var[n_pde] = np.zeros(self.N_mc)

            self.dmr[n_pde] = np.zeros(self.N_mc)

            self.quad_mean[n_pde] = 0. # E[Q_quad]
            self.quad_var[n_pde] = 0. # E[(Q_quad)^2]
            self.quad_diff_mean[n_pde] = np.zeros(self.N_mc) # MC[Q-Q_quad]
            self.quad_fval_mean[n_pde] = np.zeros(self.N_mc) # MC[Q] same as self.Q_mc
            self.quad_diff_var[n_pde] = np.zeros(self.N_mc)  # MC[(Q)^2 - (Q_quad)^2]
            self.quad_fval_var[n_pde] = np.zeros(self.N_mc)  # MC[(Q)^2]

        self.dlcomm = self.Vh[STATE].mesh().mpi_comm()

        self.mpi_rank = dl.MPI.rank(Vh[OPTIMIZATION].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(Vh[OPTIMIZATION].mesh().mpi_comm())
        if self.mpi_size > 1:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD

        if self.correction:
            self.mc_mean, self.mc_var, self.m_mc, self.x_mc, self.y_mc, self.Q_mc = \
            [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde, [None]*self.N_pde
            for n_pde in range(self.N_pde):
                self.mc_mean[n_pde] = np.zeros(self.N_mc)
                self.mc_var[n_pde] = np.zeros(self.N_mc)

                self.m_mc[n_pde] = []
                self.x_mc[n_pde] = []
                self.y_mc[n_pde] = []
                self.Q_mc[n_pde] = np.zeros(self.N_mc)

                randomGen = Random(myid=0, nproc=self.mpi_size)
                for i in range(self.N_mc):
                    noise = dl.Vector()
                    sample = dl.Vector()
                    prior.init_vector(noise, "noise")
                    prior.init_vector(sample, 1)

                    randomGen.normal(1., noise)

                    if i == 0:  # add mean in the random samples
                        noise.zero()

                    prior.sample(noise, sample, add_mean=True)
                    self.m_mc[n_pde].append(sample)

                    self.x_mc[n_pde].append(pde[n_pde].generate_state())
                    self.y_mc[n_pde].append(pde[n_pde].generate_state())

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


    def objective(self):

        cost = 0.
        for n_pde in range(self.N_pde):
            # solve the forward problem at the mean
            self.x_all[n_pde][OPTIMIZATION] = self.z
            self.pde[n_pde].solveFwd(self.x[n_pde], self.x_all[n_pde], self.tol)
            Q_0 = self.qoi[n_pde].eval(self.x[n_pde]) # Q(x(m_0))
            self.x_all[n_pde][STATE] = self.x[n_pde]

            # solve the adjoint problem at the mean
            rhs = self.pde[n_pde].generate_state()
            self.qoi[n_pde].adj_rhs(self.x[n_pde], rhs)
            self.pde[n_pde].solveAdj(self.y[n_pde], self.x_all[n_pde], rhs, self.tol)
            self.x_all[n_pde][ADJOINT] = self.y[n_pde]

            # set the linearization point for incremental forward and adjoint problems
            self.pde[n_pde].setLinearizationPoint(self.x_all[n_pde])
            self.qoi[n_pde].setLinearizationPoint(self.x[n_pde])

            # E[Q_lin]
            self.lin_mean[n_pde] = Q_0
            # beta*Var[Q_lin]
            x_fun = vector2Function(self.x[n_pde], self.Vh[STATE])
            y_fun = vector2Function(self.y[n_pde], self.Vh[ADJOINT])
            m_fun = vector2Function(self.m, self.Vh[PARAMETER])
            z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])

            f_form = self.pde[n_pde].residual(x_fun, m_fun, y_fun, z_fun, data=self.pde[n_pde].data)
            m_test = dl.TestFunction(self.Vh[PARAMETER])
            dmq = dl.assemble(dl.derivative(f_form,m_fun,m_test))
            Cdmq = self.pde[n_pde].generate_parameter()
            self.prior.Rsolver.solve(Cdmq, dmq)
            self.Cdmq[n_pde] = Cdmq
            self.lin_var[n_pde] = Q_0**2 + dmq.inner(Cdmq)

            # Monte Carlo corrections
            if self.correction:
                for i in range(self.N_mc):
                    # solve the forward problem at the new sample
                    m = self.m_mc[n_pde][i]
                    m_i = self.pde[n_pde].generate_parameter()
                    m_i.axpy(1., m)
                    m_i.axpy(1., self.m)
                    x_all = [self.x[n_pde], m_i, self.y[n_pde], self.z]
                    x = self.pde[n_pde].generate_state()
                    self.pde[n_pde].solveFwd(x, x_all, self.tol)
                    self.x_mc[n_pde][i] = x
                    Q_i = self.qoi[n_pde].eval(x) #Q(x(m_i))
                    self.Q_mc[n_pde][i] = Q_i

                    f_form = self.pde[n_pde].residual(x_fun, m_fun, y_fun, z_fun, data=self.pde[n_pde].data)
                    m_hat = vector2Function(self.m_mc[n_pde][i], self.Vh[PARAMETER])
                    dmr = dl.assemble(dl.derivative(f_form, m_fun, m_hat))
                    self.dmr[n_pde][i] = dmr

                    # Monte Carlo correction E[Q-Q_lin]
                    self.lin_diff_mean[n_pde][i] = Q_i - (Q_0+dmr)
                    self.lin_fval_mean[n_pde][i] = Q_i
                    # Monte Carlo correction E[(Q)^2 - (Q_lin)^2]
                    self.lin_diff_var[n_pde][i] = (Q_i)**2 - (Q_0+dmr)**2
                    self.lin_fval_var[n_pde][i] = (Q_i)**2

            cost += self.lin_mean[n_pde] + np.mean(self.lin_diff_mean[n_pde]) \
                         + self.beta*(self.lin_var[n_pde] + np.mean(self.lin_diff_var[n_pde])) \
                         - self.beta*(self.lin_mean[n_pde]+np.mean(self.lin_diff_mean[n_pde]))**2

            if self.mpi_rank == 0:
                print("#######################linear approximation########################")
                header = ["mean", "mean_diff", "var", "var_diff"]
                print('{:<20} {:<20} {:<20} {:<20}'.format(*header))
                data = [self.lin_mean[n_pde], np.mean(self.lin_diff_mean[n_pde]),
                        self.lin_var[n_pde]-self.lin_mean[n_pde]**2, np.mean(self.lin_diff_var[n_pde])]
                print('{:<20} {:<20} {:<20} {:<20}'.format(*data))
                print("####################################################################")

                if self.correction:
                    print ("########################################################################")
                    header = ["mean_mse", "lin_mean_diff_mse", "var_mse", "lin_var_diff_mse"]
                    print('{:<20} {:<20} {:<20} {:<20}'.format(*header))
                    data = [np.var(self.lin_fval_mean[n_pde])/self.N_mc, np.var(self.lin_diff_mean[n_pde])/self.N_mc,
                            np.var(self.lin_fval_var[n_pde])/self.N_mc, np.var(self.lin_diff_var[n_pde])/self.N_mc]
                    print('{:<20} {:<20} {:<20} {:<20}'.format(*data))
                    print ("#########################################################################")

        return cost

    def costValue(self, z):
        self.func_ncalls += 1

        if isinstance(z, np.ndarray):
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




        # if cost > 0:
        #     self.sign = 1.
        # else:
        #     self.sign = -1.
        #
        # cost*=self.sign

        if self.mpi_rank == 0:
            header = ["# func_calls", "Cost", "Objective", "Penalization"]
            print("\n {:<20} {:<20} {:<20} {:<20}".format(*header))
            data = [self.func_ncalls, cost, objective, penalization]
            print('{:<20d} {:<20.2e} {:<20.2e} {:<20.2e}'.format(*data))

        return cost

    def solve_xstar(self):
        for n_pde in range(self.N_pde):
            z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])
            x_fun = vector2Function(self.x[n_pde], self.Vh[STATE])
            y_fun = vector2Function(self.y[n_pde], self.Vh[ADJOINT])
            m_fun = vector2Function(self.m, self.Vh[PARAMETER])
            f_form = self.pde[n_pde].residual(x_fun, m_fun, y_fun, z_fun, data=self.pde[n_pde].data)
            y_test = dl.TestFunction(self.Vh[ADJOINT])
            Cdmq_fun = vector2Function(self.Cdmq[n_pde], self.Vh[PARAMETER])

            xstarrhs = self.pde[n_pde].generate_state()
            dmrCdmq = dl.derivative(f_form, m_fun, Cdmq_fun)
            dmyrCdmq = dl.assemble(dl.derivative(dmrCdmq, y_fun, y_test))
            [bc.apply(dmyrCdmq) for bc in self.pde[n_pde].bc0]
            # self.pde.bc0.apply(dmyrCdmq)
            xstarrhs.axpy(2*self.beta, dmyrCdmq)

            if self.correction:
                for i in range(self.N_mc):
                    m_hat = vector2Function(self.m_mc[n_pde][i], self.Vh[PARAMETER])
                    dmr = dl.derivative(f_form, m_fun, m_hat)
                    dmyr = dl.derivative(dmr, y_fun, y_test)
                    dmyr_ass = dl.assemble(dmyr)
                    [bc.apply(dmyr_ass) for bc in self.pde[n_pde].bc0]
                    # self.pde.bc0.apply(dmyr_ass)
                    xstarrhs.axpy(- 1./self.N_mc
                                  - 2.*self.beta/self.N_mc*(self.lin_mean[n_pde] + self.dmr[n_pde][i])
                                  + 2.*self.beta/self.N_mc*(self.lin_mean[n_pde]+np.mean(self.lin_diff_mean[n_pde])),
                                  dmyr_ass)

            xstar = self.pde[n_pde].generate_state()
            self.pde[n_pde].solver_fwd_inc.solve(xstar, -xstarrhs)
            self.xstar[n_pde] = xstar

    def solve_ystar(self):
        for n_pde in range(self.N_pde):
            z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])
            x_fun = vector2Function(self.x[n_pde], self.Vh[STATE])
            y_fun = vector2Function(self.y[n_pde], self.Vh[ADJOINT])
            m_fun = vector2Function(self.m, self.Vh[PARAMETER])
            f_form = self.pde[n_pde].residual(x_fun, m_fun, y_fun, z_fun, data=self.pde[n_pde].data)
            x_test = dl.TestFunction(self.Vh[STATE])
            Cdmq_fun = vector2Function(self.Cdmq[n_pde], self.Vh[PARAMETER])

            ystarrhs = self.pde[n_pde].generate_state()

            dxq = self.pde[n_pde].generate_state()
            # x = dl.Function(self.Vh[STATE]).vector()
            self.qoi[n_pde].grad_state(self.x[n_pde], dxq)
            [bc.apply(dxq) for bc in self.pde[n_pde].bc0]
            # self.pde.bc0.apply(dxq)
            ystarrhs.axpy(1., dxq)

            dmrCdmq = dl.derivative(f_form, m_fun, Cdmq_fun)
            dmxrCdmq = dl.assemble(dl.derivative(dmrCdmq, x_fun, x_test))
            [bc.apply(dmxrCdmq) for bc in self.pde[n_pde].bc0]
            # self.pde.bc0.apply(dmxrCdmq)
            ystarrhs.axpy(2*self.beta, dmxrCdmq)

            xstar_fun = vector2Function(self.xstar[n_pde], self.Vh[STATE])
            dxr = dl.derivative(f_form, x_fun, xstar_fun)
            dxxr = dl.derivative(dxr, x_fun, x_test)
            dxxr = dl.assemble(dxxr)
            [bc.apply(dxxr) for bc in self.pde[n_pde].bc0]
            # self.pde.bc0.apply(dxxr)
            ystarrhs.axpy(1., dxxr)

            dxxq = self.pde[n_pde].generate_state()
            self.qoi[n_pde].apply_ij(STATE, STATE, self.xstar[n_pde], dxxq)
            [bc.apply(dxxq) for bc in self.pde[n_pde].bc0]
            # self.pde.bc0.apply(dxxq)
            ystarrhs.axpy(1., dxxq)

            if self.correction:
                ystarrhs.axpy(2.*self.beta*self.lin_mean[n_pde], dxq)
                for i in range(self.N_mc):
                    m_hat = vector2Function(self.m_mc[n_pde][i], self.Vh[PARAMETER])
                    dmr = dl.derivative(f_form, m_fun, m_hat)
                    dmxr = dl.derivative(dmr, x_fun, x_test)
                    dmxr_ass = dl.assemble(dmxr)
                    [bc.apply(dmxr_ass) for bc in self.pde[n_pde].bc0]
                    # self.pde.bc0.apply(dmxr_ass)
                    ystarrhs.axpy(- 1./self.N_mc
                                  - 2.*self.beta/self.N_mc*(self.lin_mean[n_pde]+self.dmr[n_pde][i]),
                                  dxq)
                    ystarrhs.axpy(- 1./self.N_mc
                                  - 2.*self.beta/self.N_mc*(self.lin_mean[n_pde]+self.dmr[n_pde][i])
                                  + 2.*self.beta/self.N_mc*(self.lin_mean[n_pde]+np.mean(self.lin_diff_mean[n_pde])),
                                  dmxr_ass)

            ystar = self.pde[n_pde].generate_state()
            self.pde[n_pde].solver_adj_inc.solve(ystar, -ystarrhs)
            self.ystar[n_pde] = ystar

    def solve_ymc(self):
        for n_pde in range(self.N_pde):
            for i in range(self.N_mc):
                m = self.m_mc[n_pde][i]
                m_i = self.pde[n_pde].generate_parameter()
                m_i.axpy(1., m)
                m_i.axpy(1.,self.m)
                x_all = [self.x[n_pde], m_i, self.y[n_pde], self.z]
                rhs = self.pde[n_pde].generate_state()
                self.qoi[n_pde].adj_rhs(self.x_mc[n_pde][i], rhs)
                [bc.apply(rhs) for bc in self.pde[n_pde].bc0]

                rhs[:] *= 1./self.N_mc*(1.
                                        + 2.*self.beta*(self.lin_fval_mean[n_pde][i])
                                        - 2.*self.beta*(self.lin_mean[n_pde]+np.mean(self.lin_diff_mean[n_pde])))

                y_mc = self.pde[n_pde].generate_state()
                self.pde[n_pde].solveAdj(y_mc, x_all, rhs, self.tol)
                self.y_mc[n_pde][i] = y_mc

    def costGradient(self, z):
        self.grad_ncalls += 1
        self.cg_ncalls.append(self.ncalls)
        self.ncalls = 0

        if isinstance(z, np.ndarray):
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

        if self.correction:
            self.solve_ymc()
        self.solve_xstar()
        self.solve_ystar()

        if self.mpi_rank == 0:
            print ("#####################################################################")
            print (time.time() - tgrad, "seconds for solving adjoint problems")
            print ("#####################################################################")

        dzq = self.generate_optimization()
        z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])

        for n_pde in range(self.N_pde):
            x_fun = vector2Function(self.x[n_pde], self.Vh[STATE])
            y_fun = vector2Function(self.y[n_pde], self.Vh[ADJOINT])
            m_fun = vector2Function(self.m, self.Vh[PARAMETER])
            f_form = self.pde[n_pde].residual(x_fun, m_fun, y_fun, z_fun, data=self.pde[n_pde].data)
            Cdmq_fun = vector2Function(self.Cdmq[n_pde], self.Vh[PARAMETER])

            dmrCdmq = dl.derivative(f_form, m_fun, Cdmq_fun)
            dmzrCdmq = dl.assemble(dl.derivative(dmrCdmq, z_fun, z_test))
            dzq.axpy(2*self.beta, dmzrCdmq)

            if self.correction:
                for i in range(self.N_mc):
                    m_hat = vector2Function(self.m_mc[n_pde][i], self.Vh[PARAMETER])
                    dmr = dl.derivative(f_form, m_fun, m_hat)
                    dmzr = dl.derivative(dmr, z_fun, z_test)
                    dmzr_ass = dl.assemble(dmzr)

                    dzq.axpy(- 1./self.N_mc
                            - 2.*self.beta/self.N_mc*(self.lin_mean[n_pde]+self.dmr[n_pde][i])
                            + 2.*self.beta/self.N_mc*(self.lin_mean[n_pde]+np.mean(self.lin_diff_mean[n_pde])),
                            dmzr_ass)

            ystar_fun = vector2Function(self.ystar[n_pde], self.Vh[ADJOINT])
            dyr = dl.derivative(f_form, y_fun, ystar_fun)
            dyzr = dl.derivative(dyr, z_fun, z_test)
            dyzr = dl.assemble(dyzr)

            dzq.axpy(1., dyzr)

            xstar_fun = vector2Function(self.xstar[n_pde], self.Vh[STATE])
            dxr = dl.derivative(f_form, x_fun, xstar_fun)
            dxzr = dl.derivative(dxr, z_fun, z_test)
            dxzr = dl.assemble(dxzr)

            dzq.axpy(1., dxzr)

            if self.correction:
                for i in range(self.N_mc):
                    m = self.m_mc[n_pde][i]
                    m_i = self.pde[n_pde].generate_parameter()
                    m_i.axpy(1., m)
                    m_i.axpy(1., self.m)
                    m_fun = vector2Function(m_i, self.Vh[PARAMETER])
                    x_fun = vector2Function(self.x_mc[n_pde][i], self.Vh[STATE])
                    y_fun = vector2Function(self.y_mc[n_pde][i], self.Vh[ADJOINT])
                    f_form = self.pde[n_pde].residual(x_fun, m_fun, y_fun, z_fun, data=self.pde[n_pde].data)

                    dyr = dl.derivative(f_form, y_fun, y_fun)
                    dyzr = dl.derivative(dyr, z_fun, z_test)
                    dyzr = dl.assemble(dyzr)

                    dzq.axpy(1., dyzr)

        # ## contribution from the regularization term
        # chi = dl.Expression("((pow(x[0]-10.,2)+pow(x[1]-5.,2))>=4)*((pow(x[0]-10.,2)+pow(x[1]-5.,2))<=16)")
        # dzq.axpy(2.*self.alpha, dl.assemble(dl.inner(self.z,z_test)*chi*dl.dx))

        dz = self.pde[n_pde].generate_optimization()
        dz.axpy(1., dzq)

        dzp = self.pde[n_pde].generate_optimization()
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
        if isinstance(z, np.ndarray):
            dz = dz.gather_on_zero()
            if self.mpi_size > 1:
                dz = self.comm.bcast(dz, root=0)

        self.dz = dz

        tmp = self.generate_optimization()
        self.Msolver.solve(tmp, dz)
        dznorm = np.sqrt(dz.inner(tmp))

        return dz, dznorm

    def costHessian(self, z, z_dir, FD=True):
        self.hess_ncalls += 1
        self.ncalls += 1

        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0]:idx[1]])
        else:
            self.z.zero()
            self.z.axpy(1.0, z)
        if isinstance(z_dir, np.ndarray):
            idx = self.z_dir.local_range()
            self.z_dir.set_local(z_dir[idx[0]:idx[1]])
        else:
            self.z_dir.zero()
            self.z_dir.axpy(1.0, z_dir)

        if FD:
            epsilon = 1.e-6
            if isinstance(z, np.ndarray):
                z1 = z - epsilon*z_dir
                z2 = z + epsilon*z_dir
            else:
                z1 = z.axpy(-epsilon, z_dir)
                z2 = z.axpy(epsilon, z_dir)

            self.costValue(z1)
            dz1 = self.costGradient(z1)
            self.costValue(z2)
            dz2 = self.costGradient(z2)

            if isinstance(z, np.ndarray):
                Hz = (dz2-dz1)/(2*epsilon)
            else:
                Hz = dz2.axpy(-1.0, dz1)/(2*epsilon)

            # z2 = z + epsilon*z_dir
            # self.costValue(z2)
            # dz2 = self.costGradient(z2)
            # Hz_nparray = (dz2-self.dz)/epsilon

            # self.costValue(z)
            # self.costGradient(z)

            if self.mpi_rank == 0:
                print("hess_ncalls = ", self.hess_ncalls)

            return Hz

        else:
            thess = time.time()
            Hz = self.generate_optimization()
            for n_pde in range(self.N_pde):
                z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])
                z_dir_fun = vector2Function(self.z_dir, self.Vh[OPTIMIZATION])
                m_fun = vector2Function(self.m, self.Vh[PARAMETER])
                x_fun = vector2Function(self.x[n_pde], self.Vh[STATE])
                y_fun = vector2Function(self.y[n_pde], self.Vh[ADJOINT])

                r_form = self.pde[n_pde].residual(x_fun, m_fun, y_fun, z_fun, data=self.pde[n_pde].data)

                # solve incremental state
                x_trial = dl.TrialFunction(self.Vh[STATE])
                x_star = dl.Function(self.Vh[STATE])
                y_test = dl.TestFunction(self.Vh[ADJOINT])

                rx_form = dl.derivative(r_form, x_fun, x_trial)
                rxy_form = dl.derivative(rx_form, y_fun, y_test)

                rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
                rzy_form = dl.derivative(rz_form, y_fun, y_test)
                Ly_form = -rzy_form

                dl.solve(rxy_form == Ly_form, x_star, self.pde[n_pde].bc0)  # specify Kylov solver

                # solve incremental adjoint
                x_test = dl.TestFunction(self.Vh[STATE])
                y_trial = dl.TrialFunction(self.Vh[ADJOINT])
                y_star = dl.Function(self.Vh[ADJOINT])

                ry_form = dl.derivative(r_form, y_fun, y_trial)
                ryx_form = dl.derivative(ry_form, x_fun, x_test)

                rx_form = dl.derivative(r_form, x_fun, x_star)
                rxx_form = dl.derivative(rx_form, x_fun, x_test)

                qx_form = dl.derivative(self.qoi[n_pde].form(x_fun), x_fun, x_star)
                qxx_form = dl.derivative(qx_form, x_fun, x_test)

                rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
                rzx_form = dl.derivative(rz_form, x_fun, x_test)

                Lx_form = -(rxx_form + qxx_form + rzx_form)

                dl.solve(ryx_form == Lx_form, y_star, self.pde[n_pde].bc0)

                # assemble Hessian
                z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
                ry_form = dl.derivative(r_form, y_fun, y_star)
                ryz_form = dl.derivative(ry_form, z_fun, z_test)

                rx_form = dl.derivative(r_form, x_fun, x_star)
                rxz_form = dl.derivative(rx_form, z_fun, z_test)

                rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
                rzz_form = dl.derivative(rz_form, z_fun, z_test)

                Lz_form = ryz_form + rxz_form + rzz_form

                Hz.axpy(1.0, dl.assemble(Lz_form))

            dzzp = self.pde[0].generate_optimization()
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
        # z_trial = dl.TrialFunction(self.Vh[OPTIMIZATION])
        # z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
        # self.Hessian = dl.assemble(z_trial*z_test*dl.dx)

        self.Hessian.init_vector(z, dim)

        # znew = self.pde.generate_optimization()
        # z.init(znew.size())

    def mult(self, zhat, Hzhat, FD=False):

        z = self.z.copy()

        Hz = self.costHessian(z, zhat, FD=FD)

        Hzhat.zero()
        Hzhat.axpy(1.0, Hz)