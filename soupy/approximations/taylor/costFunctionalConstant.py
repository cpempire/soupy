
from __future__ import absolute_import, division, print_function

import time
import numpy as np
import dolfin as dl
dl.dx = dl.dx(metadata={'quadrature_degree':2, "representation":'uflacs'}) #, "representation":'uflacs'

# import sys
# sys.path.append('../')
from ..costFunctional import CostFunctional
# sys.path.append('../../')
from ...utils.variables import STATE, PARAMETER, ADJOINT, OPTIMIZATION
from ...utils.vector2function import vector2Function
from ...utils.checkDolfinVersion import dlversion

class CostFunctionalConstant(CostFunctional):
    """
    The objective is E[Q] + beta Var[Q] = Q(u(m_0)), by Taylor zeroth approximation

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
        self.quad_fval_mean = np.zeros(self.N_mc) # MC[Q] same as self.Q_mc
        self.quad_diff_var = np.zeros(self.N_mc)  # MC[(Q-Q_0)^2 - (Q_quad-Q_0)^2]
        self.quad_fval_var = np.zeros(self.N_mc)  # MC[(Q-Q_0)^2]

        self.dlcomm = self.Vh[STATE].mesh().mpi_comm()

        self.mpi_rank = dl.MPI.rank(Vh[OPTIMIZATION].mesh().mpi_comm())
        self.mpi_size = dl.MPI.size(Vh[OPTIMIZATION].mesh().mpi_comm())
        if self.mpi_size > 1:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD

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
            # self.Msolver = dl.PETScKrylovSolver("cg", "jacobi")
            self.Msolver = dl.PETScLUSolver()
        else:
            # self.Msolver = dl.PETScKrylovSolver(self.Vh[OPTIMIZATION].mesh().mpi_comm(), "cg", "jacobi")
            self.Msolver = dl.PETScLUSolver(self.Vh[OPTIMIZATION].mesh().mpi_comm())

        self.Msolver.set_operator(self.Hessian)
        # self.Msolver.parameters["maximum_iterations"] = 100
        # self.Msolver.parameters["relative_tolerance"] = 1.e-12
        # self.Msolver.parameters["error_on_nonconvergence"] = True
        # self.Msolver.parameters["nonzero_initial_guess"] = False

    def objective(self):

        self.x_all[OPTIMIZATION] = self.z
        self.pde.solveFwd(self.x, self.x_all, self.tol)
        Q_0 = self.qoi.eval(self.x) # Q(x(m_0))
        self.x_all[STATE] = self.x

        # x_fun = vector2Function(x, self.Vh[STATE])
        # x_fun_r, x_fun_i = x_fun.split(True)
        # dl.plot(x_fun_r)
        # dl.plot(x_fun_i)
        # dl.interactive()

        return Q_0

    def costValue(self, z):
        self.func_ncalls += 1

        if isinstance(z, np.ndarray):
            idx = self.z.local_range()
            self.z.set_local(z[idx[0]:idx[1]])
        else:
            self.z.zero()
            self.z.axpy(1.0, z)

        # print("z = ", z)
        tobj = time.time()

        objective = self.objective()

        self.tobj = time.time()-tobj

        if self.mpi_rank == 0:
            print("#####################################################################")
            print(time.time()-tobj, "seconds for objective evaluation")
            print("#####################################################################")

        penalization = self.penalization.eval(self.z)

        cost = objective + penalization

        if self.mpi_rank == 0:
            header = ["# func_calls", "Cost", "Objective", "Penalization"]
            print("\n {:<20} {:<20} {:<20} {:<20}".format(*header))
            data = [self.func_ncalls, cost, objective, penalization]
            print('{:<20d} {:<20.2e} {:<20.2e} {:<20.2e}'.format(*data))

        # return cost, objective, penalization
        #
        return cost

    def costGradient(self, z):
        self.grad_ncalls += 1
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

        self.x_all[OPTIMIZATION] = self.z

        # solve the adjoint problem
        rhs = self.pde.generate_state()
        self.qoi.adj_rhs(self.x, rhs)
        self.pde.solveAdj(self.y, self.x_all, rhs, self.tol)
        self.x_all[ADJOINT] = self.y

        # compute the gradient
        dz = self.pde.generate_optimization()

        dzq = self.pde.generate_optimization()
        self.pde.evalGradientControl(self.x_all, dzq)
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

        tmp = self.generate_optimization()
        self.Msolver.solve(tmp, dz)
        dznorm = np.sqrt(dz.inner(tmp))

        return dz, dznorm

    def costHessian(self, z, z_dir, FD=False):
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

    # # define reduced z-Hessian used for Newton CG algorithm with function Hessian.init_vector and Hessian.mult
    # def init_vector(self, z, dim):
    #     z_trial = dl.TrialFunction(self.Vh[OPTIMIZATION])
    #     z_test = dl.TestFunction(self.Vh[OPTIMIZATION])
    #     self.Hessian = dl.assemble(z_trial*z_test*dl.dx)

    #     self.Hessian.init_vector(z, dim)
    #
    #     # znew = self.pde.generate_optimization()
    #     # z.init(znew.size())
    #
    # def mult(self, zhat, Hzhat, FD=False):
    #
    #     z = self.z.copy()
    #
    #     Hz = self.costHessian(z, zhat, FD=FD)
    #
    #     Hzhat.zero()
    #     Hzhat.axpy(1.0, Hz)