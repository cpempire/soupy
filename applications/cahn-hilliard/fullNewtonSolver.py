from __future__ import absolute_import, division, print_function

import math
path = "../../"
import sys
sys.path.append(path)
from soupy import *


def LS_ParameterList():
    """
    Generate a ParameterList for line search globalization.
    type: :code:`LS_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["c_armijo"] = [1e-4, "Armijo constant for sufficient reduction"]
    parameters["max_backtracking_iter"] = [0, "Maximum number of backtracking iterations"]

    return ParameterList(parameters)


def TR_ParameterList():
    """
    Generate a ParameterList for Trust Region globalization.
    type: :code:`RT_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["eta"] = [0.05, "Reject step if (actual reduction)/(predicted reduction) < eta"]

    return ParameterList(parameters)


def Newton_ParameterList():
    """
    Generate a ParameterList for ReducedSpaceNewtonCG.
    type: :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["linear_solver"] = ["lu", "home_cg for CGSolverSteihaug"]
    parameters["rel_tolerance"] = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"] = [1e-18, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdm_tolerance"] = [1e-18, "we converge when (g,dm) <= gdm_tolerance"]
    parameters["maximum_iterations"] = [200, "maximum number of iterations"]
    parameters["inner_rel_tolerance"] = [1e-9,
                                         "relative tolerance used for the solution of the forward, adjoint, and incremental (fwd,adj) problems"]
    parameters["globalization"] = ["LS", "Globalization technique: line search (LS)  or trust region (TR)"]
    parameters["print_level"] = [2, "Control verbosity of printing screen"]
    parameters["GN_iter"] = [5, "Number of Gauss Newton iterations before switching to Newton"]
    parameters["cg_coarse_tolerance"] = [.5, "Coarsest tolerance for the CG method (Eisenstat-Walker)"]
    parameters["LS"] = [LS_ParameterList(), "Sublist containing LS globalization parameters"]
    parameters["TR"] = [TR_ParameterList(), "Sublist containing TR globalization parameters"]

    return ParameterList(parameters)


class FullNewtonSolver:
    """
    A Newton solver to solve nonlinear PDE problems.
    The Newton system is solved either by LU solver or
    inexactly by early termination of CG iterations via Eisenstat-Walker
    (to prevent oversolving) and Steihaug (to avoid negative curvature) criteria.
    Globalization is performed using one of the following methods:

    - line search (LS) based on the armijo sufficient reduction condition; or
    - trust region (TR) based on the prior preconditioned norm of the update direction.

    The stopping criterion is based on a control on the norm of the gradient and a control of the
    inner product between the gradient and the Newton direction.

    """
    termination_reasons = [
        "Maximum number of Iteration reached",  # 0
        "Norm of the gradient less than tolerance",  # 1
        "Maximum number of backtracking reached",  # 2
        "Norm of (g, dm) less than tolerance"  # 3
        ]

    def __init__(self, problem, parameters=Newton_ParameterList()):
        """
        Initialize the ReducedSpaceNewtonCG.
        Type :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for list of default parameters
        and their descriptions.
        """
        self.problem = problem
        self.mpi_rank = self.problem.mpi_rank
        self.parameters = parameters
        self.it = 0
        self.converged = False
        self.total_cg_iter = 0
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0

        self.Iteration = []
        self.costValue = []
        self.costGrad = []
        self.ItLS = []

        if self.parameters["linear_solver"] is "home_cg":
            from .cgsolverSteihaug import CGSolverSteihaug
            self.solver = CGSolverSteihaug()
            self.solver.parameters["maximum_iterations"] = 20
            self.solver.parameters["zero_initial_guess"] = True
            self.solver.parameters["print_level"] = self.parameters["print_level"] - 1
        else:
            self.solver = self.problem.solver

    def solve(self, z):
        """

        Input:
            :code:`z` represents the initial guess.
            :code:`z` will be overwritten on return.
        """

        if self.parameters["globalization"] == "LS":
            return self._solve_ls(z)
        elif self.parameters["globalization"] == "TR":
            return self._solve_tr(z)
        else:
            raise ValueError(self.parameters["globalization"])

    def _solve_ls(self, z):
        """
        Solve the nonliear problem with initial guess :code:`z`.
        """
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        maximum_iterations = self.parameters["maximum_iterations"]
        innerTol = self.parameters["inner_rel_tolerance"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]

        c_armijo = self.parameters["LS"]["c_armijo"]
        max_backtracking_iter = self.parameters["LS"]["max_backtracking_iter"]

        self.it = 0
        self.converged = False
        self.ncalls += 1

        zhat = self.problem.generate_vector()
        dz = self.problem.generate_vector()
        z_star = self.problem.generate_vector()

        cost_old = self.problem.cost(z)
        cost_new = cost_old
        self.costValue.append(cost_old)

        if self.mpi_rank == 0:
            f = open("iterate.dat", "a+")
            f.write("\n {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "It", "cost", "(g,dm)", "||g||L2", "alpha"))

        if self.problem.save:
            self.problem.z_save.vector().set_local(z.get_local())
            self.problem.file_u << (self.problem.z_save.split()[0], 0.)
            # self.problem.file_mu << (self.problem.z_save.split()[1], 0.)

        while (self.it < maximum_iterations) and (self.converged is False):

            dz, gradnorm = self.problem.gradient(z)
            self.costGrad.append(gradnorm)

            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini * rel_tol)

            A = self.problem.A(z)
            self.solver.set_operator(A)

            if self.parameters["linear_solver"] is "home_cg":
                tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm / gradnorm_ini))
                self.solver.set_preconditioner(self.problem.P)
                self.solver.parameters["rel_tolerance"] = tolcg

                self.solver.solve(zhat, -dz)
                # self.total_cg_iter += self.problem.A.ncalls
            else:
                self.solver.solve(zhat, -dz)

            dz_zhat = dz.inner(zhat)

            # modify Hessian to make it positive definite
            if dz_zhat > 0 and self.problem.rescaling:
                print("modify Hessian for a descent direction")
                A = self.problem.A_modified(z)
                self.solver.set_operator(A)
                self.solver.solve(zhat, -dz)
                dz_zhat = dz.inner(zhat)

            alpha = 1.0
            descent = 0
            n_backtrack = 0
            start_backtracking = 40

            while self.it >= start_backtracking and descent == 0 and n_backtrack < max_backtracking_iter:
                if self.it == start_backtracking:
                    cost_old = self.problem.cost(z)

                z_star.zero()
                z_star.axpy(1., z)
                z_star.axpy(alpha, zhat)

                cost_new = self.problem.cost(z_star)
                print("n_backtrack = ", n_backtrack, "energy = ", cost_new)

                # Check if armijo conditions are satisfied
                if cost_new > 0 and ((cost_new < cost_old + alpha * c_armijo * dz_zhat) or (
                        -dz_zhat <= self.parameters["gdm_tolerance"])):
                    cost_old = cost_new
                    self.costValue.append(cost_old)
                    descent = 1
                    z.zero()
                    z.axpy(1., z_star)
                else:
                    n_backtrack += 1
                    alpha *= 0.5

            if self.it < start_backtracking or max_backtracking_iter == 0:
                z.axpy(1., zhat)
                cost_new = self.problem.cost(z)

            if self.problem.save:
                self.problem.z_save.vector().set_local(z.get_local())
                self.problem.file_u << (self.problem.z_save.split()[0], self.it+1.)
                # self.problem.file_mu << (self.problem.z_save.split()[1], self.it+1.)

            self.ItLS.append(n_backtrack)

            if self.mpi_rank == 0:
                if (print_level >= 0):
                    print("\n {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                        "It", "cost", "(g,dm)", "||g||L2", "alpha"))
                    print("{:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                        self.it, cost_new, dz_zhat, gradnorm/gradnorm_ini, alpha))
                f.write(" \n {:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                    self.it, cost_new, dz_zhat, gradnorm/gradnorm_ini, alpha))

            self.it += 1

            if self.it > 5:
                # check if solution is reached
                if (gradnorm < tol) and (self.it > 0):
                    self.converged = True
                    self.reason = 1
                    break

                if max_backtracking_iter > 0 and n_backtrack == max_backtracking_iter:
                    self.converged = False
                    self.reason = 2
                    break

                if -dz_zhat <= self.parameters["gdm_tolerance"] and self.problem.rescaling:
                    self.converged = True
                    self.reason = 3
                    break

        self.final_grad_norm = gradnorm
        self.final_cost = cost_new

        if self.mpi_rank == 0:
            print(self.termination_reasons[self.reason])
            f.write("\n\n" + "Termination reason: " + self.termination_reasons[self.reason] + "\n\n\n")
            f.close()

        result = dict()
        result["Iteration"] = self.Iteration
        result["costValue"] = self.costValue
        result["costGrad"] = self.costGrad
        result["ItLS"] = self.ItLS

        return result