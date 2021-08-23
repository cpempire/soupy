from __future__ import absolute_import, division, print_function

import time
import numpy as np
import dolfin as dl
dl.dx = dl.dx(metadata={'quadrature_degree': 2}) #, "representation":'uflacs'

# import sys
# sys.path.append('../')
from ..costFunctional import CostFunctional
from ...utils.vector2function import vector2Function
from ...utils.variables import STATE, PARAMETER, ADJOINT, OPTIMIZATION
from ...utils.random import Random
from ...utils.checkDolfinVersion import dlversion

class ChanceConstraintSAA(CostFunctional):
    """
    The objective is E[Q] + beta Var[Q] = Q(u(m_0)), by Taylor zeroth approximation

    The penalization term is P(z) = alpha*||z||_2^2 + ...

    The cost functional is objective + regularization
    """

    def __init__(self, parameter, Vh, pde, qoi, qoi_constraint, prior, penalization, tol=1e-9):

        self.parameter = parameter
        self.Vh = Vh
        self.pde = pde
        self.generate_optimization = pde.generate_optimization
        self.qoi = qoi
        self.qoi_constraint = qoi_constraint
        self.prior = prior
        self.penalization = penalization
        self.z = pde.generate_optimization()
        self.tol = tol

        # self.m = prior.mean
        # self.x = pde.generate_state()
        # self.y = pde.generate_state()
        # self.x_all = [self.x, self.m, self.y, self.z]

        self.z_dir = pde.generate_optimization()
        self.z_diff = pde.generate_optimization()
        self.dz = None

        self.func_ncalls = 0
        self.grad_ncalls = 0
        self.hess_ncalls = 0
        self.ncalls = 0

        self.correction = parameter["correction"]
        self.N_mc = parameter["N_mc"]
        self.beta = parameter["beta"]

        self.lin_mean = 0. # E[Q_lin]
        self.lin_var = 0. # E[(Q_lin-Q_0)^2]
        self.lin_diff_mean = np.zeros(self.N_mc)
        self.lin_fval_mean = np.zeros(self.N_mc)
        self.lin_diff_var = np.zeros(self.N_mc)
        self.lin_fval_var = np.zeros(self.N_mc)

        self.quad_mean = 0. # E[Q_quad]
        self.quad_var = 0. # E[(Q_quad-Q_0)^2]
        self.quad_diff_mean = np.zeros(self.N_mc) # MC[Q-Q_quad]
        self.quad_fval_mean = np.zeros(self.N_mc) # MC[Q] same as self.Q
        self.quad_diff_var = np.zeros(self.N_mc)  # MC[(Q-Q_0)^2 - (Q_quad-Q_0)^2]
        self.quad_fval_var = np.zeros(self.N_mc)  # MC[(Q-Q_0)^2]

        self.dlcomm = self.Vh[STATE].mesh().mpi_comm()

        self.mpi_rank = dl.MPI.rank(Vh[OPTIMIZATION].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(Vh[OPTIMIZATION].mesh().mpi_comm())
        if self.mpi_size > 1:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD

        self.mean = np.zeros(self.N_mc)
        self.var = np.zeros(self.N_mc)

        self.m = []
        self.x = []
        self.y = []

        self.Q = np.zeros(self.N_mc)
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
            self.m.append(sample)

            self.x.append(pde.generate_state())
            self.y.append(pde.generate_state())

        self.tobj = 0.
        self.tgrad = 0.
        self.trand = 0.

        if 'optType' in parameter and parameter['optType'] == 'vector':
            pass
        else:
            if 'dim' in parameter:  # number of optimization variables
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

            else:  # only one field optimization variable
                z_trial = dl.TrialFunction(self.Vh[OPTIMIZATION])
                z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
                self.Hessian = dl.assemble(z_trial * z_test * dl.dx)

            if dlversion() <= (1,6,0):
                self.Msolver = dl.PETScKrylovSolver("cg", "jacobi")
            else:
                self.Msolver = dl.PETScKrylovSolver(self.Vh[OPTIMIZATION].mesh().mpi_comm(), "cg", "jacobi")
            self.Msolver.set_operator(self.Hessian)
            self.Msolver.parameters["maximum_iterations"] = 100
            self.Msolver.parameters["relative_tolerance"] = 1.e-12
            self.Msolver.parameters["error_on_nonconvergence"] = True
            self.Msolver.parameters["nonzero_initial_guess"] = False

        # for chance constraint
        self.N_cc = parameter["N_cc"]
        self.c_alpha = parameter["c_alpha"]
        self.c_beta = parameter["c_beta"]
        self.c_gamma = parameter["c_gamma"]

        if self.c_gamma > 0:

            self.chance = None
            self.f_mc = np.zeros(self.N_mc)
            self.c_f = np.zeros(self.N_mc)
            self.d_f = 0
            self.m_f = pde.generate_parameter()

    def objective(self):
        # sample average approximation
        for i in range(self.N_mc):
            x_all = [self.x[i], self.m[i], self.y[i], self.z]
            x = self.pde.generate_state()
            self.pde.solveFwd(x, x_all, self.tol)
            self.x[i] = x
            x_all[STATE] = x
            Q_i = self.qoi.eval(x_all) #Q(x(m_i))
            self.Q[i] = Q_i

            self.mean[i] = Q_i
            self.var[i] = Q_i**2

        cost = np.mean(self.mean) \
                     + self.beta*np.mean(self.var) - self.beta*(np.mean(self.mean))**2

        return cost

    def constraint(self):

        for i in range(self.N_mc):
            x_all = [self.x[i], self.m[i], self.y[i], self.z]
            self.f_mc[i] = self.qoi_constraint.eval(x_all)

        chance = self.c_alpha

        for i in range(self.N_mc):
            l_beta = 1/(1+np.exp(-2*self.c_beta*self.f_mc[i]))
            chance -= 1./self.N_mc * l_beta

            # header = ["f_0", "f_1", "f_2", "f_taylor", "l_beta"]
            # print("\n {:<20} {:<20} {:<20} {:<20} {:<20}".format(*header))
            # data = [f_0, f_1, f_2, f_taylor, l_beta]
            # print('{:<20e} {:<20.2e} {:<20.2e} {:<20.2e} {:<20.2e}'.format(*data))

        self.chance = chance

        s_gamma = self.c_gamma / 2 * np.max([0, chance])**2

        return s_gamma

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

        # treat chance constraint as penalty
        constraint = 0.
        if self.c_gamma > 0:
            constraint = self.constraint()
            cost += constraint

        if self.mpi_rank == 0:
            header = ["# func_calls", "Cost", "Objective", "Penalization", "Constraint"]
            print("\n {:<20} {:<20} {:<20} {:<20} {:<20}".format(*header))
            data = [self.func_ncalls, cost, objective, penalization, constraint]
            print('{:<20d} {:<20.2e} {:<20.2e} {:<20.2e} {:<20.2e}'.format(*data))

        self.cost = cost

        return cost

    def solve_ymc(self):
        for i in range(self.N_mc):
            x_all = [self.x[i], self.m[i], self.y[i], self.z]
            rhs = self.pde.generate_state()
            self.qoi.adj_rhs(x_all, rhs)
            [bc.apply(rhs) for bc in self.pde.bc0]
            rhs[:] *= 1./self.N_mc*(1. + 2.*self.beta*self.mean[i] - 2.*self.beta*np.mean(self.mean))

            if self.c_gamma > 0:
                # with constraint
                rhs_constraint = self.pde.generate_state()
                self.qoi_constraint.adj_rhs(x_all, rhs_constraint)
                [bc.apply(rhs_constraint) for bc in self.pde.bc0]
                rhs.axpy(self.d_f, rhs_constraint)

            self.pde.solveAdj(self.y[i], x_all, rhs, self.tol)

    def costGradient(self, z):
        self.grad_ncalls += 1
        self.ncalls = 0

        if isinstance(z, np.ndarray):
            zprev = self.z.gather_on_zero()
            if self.mpi_size > 1:
                zprev = self.comm.bcast(zprev, root=0)

            if np.linalg.norm(z-zprev) > 1.e-15:
                self.costValue(z)

            idx = self.z.local_range()
            self.z.set_local(z[idx[0]:idx[1]])
        else:
            self.z_diff.zero()
            self.z_diff.axpy(1.0, self.z)
            self.z_diff.axpy(-1.0, z)
            if self.z_diff.inner(self.z_diff) > 1.e-20:
                self.costValue(z)

            self.z.zero()
            self.z.axpy(1.0, z)

        # compute constants
        if self.c_gamma > 0:
            self.constraint()
            self.d_f = 0
            for j in range(self.N_mc):
                ds_gamma = self.c_gamma * np.max([0, self.chance])
                dl_beta = 2 * self.c_beta * np.exp(-2 * self.c_beta * self.f_mc[j]) / \
                          (1 + np.exp(-2 * self.c_beta * self.f_mc[j])) ** 2
                self.d_f += 1. / self.N_mc * ds_gamma * dl_beta

        tgrad = time.time()

        self.solve_ymc()

        if self.mpi_rank == 0:
            print("#####################################################################")
            print(time.time() - tgrad, "seconds for solving adjoint problems")
            print("#####################################################################")

        dzq = self.pde.generate_optimization()

        for i in range(self.N_mc):
            x_all = [self.x[i], self.m[i], self.y[i], self.z]
            pdzq = self.pde.generate_optimization()
            self.qoi.grad_optimization(x_all,pdzq)
            dzq.axpy(1.0/self.N_mc + 2*self.beta/self.N_mc*self.qoi.eval(x_all) - 2*self.beta*np.mean(self.Q), pdzq)

            if self.c_gamma > 0:
                pdzf = self.pde.generate_optimization()
                self.qoi_constraint.grad_optimization(x_all, pdzf)
                dzq.axpy(self.d_f, pdzf)

        z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
        z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])

        for i in range(self.N_mc):
            m_fun = vector2Function(self.m[i], self.Vh[PARAMETER])
            x_fun = vector2Function(self.x[i], self.Vh[STATE])
            y_fun = vector2Function(self.y[i], self.Vh[ADJOINT])
            f_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)

            dyr = dl.derivative(f_form, y_fun, y_fun)
            dyzr = dl.derivative(dyr, z_fun, z_test)
            dyzr = dl.assemble(dyzr)

            dzq.axpy(1., dyzr)

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
        if isinstance(z, np.ndarray):
            dz = dz.gather_on_zero()
            if self.mpi_size > 1:
                dz = self.comm.bcast(dz, root=0)

        self.dz = dz

        if isinstance(z, np.ndarray):
            self.save_iterate(z, dz)
            return dz
        else:
            tmp = self.generate_optimization()
            self.Msolver.solve(tmp, dz)
            dznorm = np.sqrt(dz.inner(tmp))

            return dz, dznorm

    def save_iterate(self, z, dz):
        if self.mpi_rank == 0:
            f = open("iteration.dat", 'a+')
            if self.grad_ncalls == 1:
                f.write("\n {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "It", "cost", "(g,dm)", "||g||L2", "alpha", "theta"))

            if 'bounds' in self.parameter:
                dznorm = 0.
                bounds = self.parameter["bounds"]
                for i in range(self.parameter["optDimension"]):
                    if dz[i] > 0:
                        dist = z[i] - bounds[i][0]
                    else:
                        dist = bounds[i][1] - z[i]
                    dznorm = np.max([dznorm, np.min([dist, np.abs(dz[i])])])
            else:
                dznorm = np.abs(dz)

            print("\n {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "It", "cost", "(g,dm)", "||g||L2", "alpha", "theta"))
            print("{:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                self.grad_ncalls, self.cost, 0., dznorm, 1.0, 1.0))
            f.write(" \n {:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                self.grad_ncalls, self.cost, 0., dznorm, 1.0, 1.0))
            f.close()

    def costHessian(self, z, z_dir, FD=False):
        self.hess_ncalls += 1
        self.ncalls += 1

        if isinstance(z,np.ndarray):
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

            m_fun = vector2Function(self.m[0], self.Vh[PARAMETER])
            x_fun = vector2Function(self.x[0], self.Vh[STATE])
            y_fun = vector2Function(self.y[0], self.Vh[ADJOINT])

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

            # #### full Hessian from all samples
            # z_fun = vector2Function(self.z, self.Vh[OPTIMIZATION])
            # z_dir_fun = vector2Function(self.z_dir, self.Vh[OPTIMIZATION])

            # # solve incremental state
            # xstar_set = []
            # qstar_set = []
            # for i in range(self.N_mc):
            #     m_fun = vector2Function(self.m[i], self.Vh[PARAMETER])
            #     x_fun = vector2Function(self.x[i], self.Vh[STATE])
            #     y_fun = vector2Function(self.y[i], self.Vh[ADJOINT])
            #     r_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)
            #
            #     x_trial = dl.TrialFunction(self.Vh[STATE])
            #     x_star = dl.Function(self.Vh[STATE])
            #     y_test = dl.TestFunction(self.Vh[ADJOINT])
            #
            #     rx_form = dl.derivative(r_form, x_fun, x_trial)
            #     rxy_form = dl.derivative(rx_form, y_fun, y_test)
            #
            #     rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
            #     rzy_form = dl.derivative(rz_form, y_fun, y_test)
            #     Ly_form = -rzy_form
            #
            #     dl.solve(rxy_form == Ly_form, x_star, self.pde.bc0) # specify Kylov solver
            #     xstar_set.append(x_star)
            #
            #     q_star_form = dl.derivative(self.qoi.form(x_fun), x_fun, x_star)
            #     q_star = dl.assemble(q_star_form)
            #     qstar_set.append(q_star)
            #
            # # solve incremental adjoint
            # ystar_set = []
            # for i in range(self.N_mc):
            #     m_fun = vector2Function(self.m[i], self.Vh[PARAMETER])
            #     x_fun = vector2Function(self.x[i], self.Vh[STATE])
            #     y_fun = vector2Function(self.y[i], self.Vh[ADJOINT])
            #     r_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)
            #
            #     x_test = dl.TestFunction(self.Vh[STATE])
            #     y_trial = dl.TrialFunction(self.Vh[ADJOINT])
            #     y_star = dl.Function(self.Vh[ADJOINT])
            #     x_star = xstar_set[i]
            #     q_star = qstar_set[i]
            #
            #     ry_form = dl.derivative(r_form, y_fun, y_trial)
            #     ryx_form = dl.derivative(ry_form, x_fun, x_test)
            #     ryx = dl.assemble(ryx_form)
            #     rhs = dl.assemble(ry_form)
            #     rhs.zero()
            #
            #     rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
            #     rzx_form = dl.derivative(rz_form, x_fun, x_test)
            #     rzx = dl.assemble(rzx_form)
            #     rhs.axpy(1.0, rzx)
            #
            #     rx_form = dl.derivative(r_form, x_fun, x_star)
            #     rxx_form = dl.derivative(rx_form, x_fun, x_test)
            #     rxx = dl.assemble(rxx_form)
            #     rhs.axpy(1.0, rxx)
            #
            #     qx_form = dl.derivative(self.qoi.form(x_fun), x_fun, x_star)
            #     qxx_form = dl.derivative(qx_form, x_fun, x_test)
            #     qxx = dl.assemble(qxx_form)
            #     rhs.axpy(1./self.N_mc * (1. + 2.*self.beta*self.Q[i] - 2.*self.beta*np.mean(self.Q)), qxx)
            #
            #     qx_form = dl.derivative(self.qoi.form(x_fun), x_fun, x_test)
            #     qx = dl.assemble(qx_form)
            #     rhs.axpy(1./self.N_mc * 2.*self.beta*(q_star - np.mean(qstar_set)), qx)
            #
            #     [bc.apply(ryx, rhs) for bc in self.pde.bc0]
            #     solver = dl.PETScLUSolver() # change to Kylov solver
            #     solver.set_operator(ryx)
            #     solver.solve(y_star.vector(), -rhs)
            #     ystar_set.append(y_star)
            #
            # # assemble Hessian
            # Hz = self.pde.generate_optimization()
            # for i in range(self.N_mc):
            #     m_fun = vector2Function(self.m[i], self.Vh[PARAMETER])
            #     x_fun = vector2Function(self.x[i], self.Vh[STATE])
            #     y_fun = vector2Function(self.y[i], self.Vh[ADJOINT])
            #     r_form = self.pde.residual(x_fun, m_fun, y_fun, z_fun)
            #
            #     z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
            #     x_star = xstar_set[i]
            #     y_star = ystar_set[i]
            #
            #     ry_form = dl.derivative(r_form, y_fun, y_star)
            #     ryz_form = dl.derivative(ry_form, z_fun, z_test)
            #
            #     rx_form = dl.derivative(r_form, x_fun, x_star)
            #     rxz_form = dl.derivative(rx_form, z_fun, z_test)
            #
            #     rz_form = dl.derivative(r_form, z_fun, z_dir_fun)
            #     rzz_form = dl.derivative(rz_form, z_fun, z_test)
            #
            #     Lz_form = ryz_form + rxz_form + rzz_form
            #     Hz.axpy(1.0, dl.assemble(Lz_form))

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