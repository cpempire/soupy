from __future__ import absolute_import, division, print_function

import numpy as np

# import sys
# sys.path.append('../')
from ..utils.parameterList import ParameterList
from .NewtonCG import LS_ParameterList

def BFGSoperator_ParameterList():
    parameters = {}
    parameters["BFGS_damping"] = [0.2, "Damping of BFGS"]
    parameters["memory_limit"] = [np.inf, "Number of vectors to store in limited memory BFGS"]
    return ParameterList(parameters)

def BFGS_ParameterList():
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdm_tolerance"]         = [1e-18, "we converge when (g,dm) <= gdm_tolerance"]
    parameters["max_iter"]              = [500, "maximum number of iterations"]
    parameters["inner_rel_tolerance"]   = [1e-9, "relative tolerance used for the solution of the forward, adjoint, and incremental (fwd,adj) problems"]
    parameters["globalization"]         = ["LS", "Globalization technique: line search (LS)  or trust region (TR)"]
    parameters["print_level"]           = [0, "Control verbosity of printing screen"]
    parameters["GN_iter"]               = [5, "Number of Gauss Newton iterations before switching to Newton"]
    parameters["cg_coarse_tolerance"]   = [.5, "Coarsest tolerance for the CG method (Eisenstat-Walker)"]
    ls_list = LS_ParameterList()
    ls_list["max_backtracking_iter"] = 25
    parameters["LS"]                    = [ls_list, "Sublist containing LS globalization parameters"]
    parameters["BFGS_op"]               = [BFGSoperator_ParameterList(), "BFGS operator"]
    return ParameterList(parameters)


class RescaledIdentity(object):
    """
    Default operator for H0inv
    Corresponds to applying d0*I
    """
    def __init__(self, init_vector=None):
        self.d0 = 1.0
        self._init_vector = init_vector

    def init_vector(self, z, dim):

        if self._init_vector:
            self._init_vector(z, dim)
        else:
            pass
            # raise NotImplementedError("an init_vector should be implemented")

    def solve(self, z, b):
        z.zero()
        z.axpy(self.d0, b)


class BFGS_operator:
    def __init__(self,  parameters=BFGSoperator_ParameterList()):
        self.S, self.Y, self.R = [], [], []

        self.H0inv = None
        self.help = None
        self.update_scaling = True

        self.parameters = parameters

    def set_H0inv(self, H0inv=None):
        """
        Set user-defined operator corresponding to H0inv
        Input:
            H0inv: Fenics operator with method 'solve'
        """
        if H0inv is not None:
            self.H0inv = H0inv
        else:
            self.H0inv = RescaledIdentity()

    def solve(self, z, b):
        """
        Solve system:           H_bfgs * z = b
        where H_bfgs is the approximation to the Hessian build by BFGS.
        That is, we apply
                                z = (H_bfgs)^{-1} * b
                                  = Hk * b
        where Hk matrix is BFGS approximation to the inverse of the Hessian.
        Computation done via double-loop algorithm.
        Inputs:
            z = vector (Fenics) [out]; z = Hk*b
            b = vector (Fenics) [in]
        """
        A = []
        if self.help is None:
            self.help = b.copy()
            # self.cscale = np.sqrt(self.help.inner(self.help))
        else:
            self.help.zero()
            self.help.axpy(1.0, b)

        for s, y, r in zip(reversed(self.S), reversed(self.Y), reversed(self.R)):
            a = r * s.inner(self.help)
            A.append(a)
            self.help.axpy(-a, y)

        # option 1
        self.H0inv.solve(z, self.help.copy())     # z = H0 * z_copy

        # # option 2
        # z.zero()
        # z.axpy(1.0/self.cscale, self.help.copy())

        for s, y, r, a in zip(self.S, self.Y, self.R, reversed(A)):
            b = r * y.inner(z)
            z.axpy(a - b, s)

    def update(self, s, y):
        """
        Update BFGS operator with most recent gradient update
        To handle potential break from secant condition, update done via damping
        Input:
            s = Vector (Fenics) [in]; corresponds to update in medium parameters
            y = Vector (Fenics) [in]; corresponds to update in gradient
        """
        damp = self.parameters["BFGS_damping"]
        memlim = self.parameters["memory_limit"]
        if self.help is None:
            self.help = y.copy()
        else:
            self.help.zero()

        sy = s.inner(y)
        self.solve(self.help, y)
        yHy = y.inner(self.help)
        theta = 1.0
        if sy < damp*yHy:
            theta = (1.0-damp)*yHy/(yHy-sy)
            s *= theta
            s.axpy(1-theta, self.help)
            sy = s.inner(y)
        assert(sy > 0.)
        rho = 1./sy
        self.S.append(s.copy())
        self.Y.append(y.copy())
        self.R.append(rho)

        # if L-BFGS
        if len(self.S) > memlim:
            self.S.pop(0)
            self.Y.pop(0)
            self.R.pop(0)
            self.update_scaling = True

        # re-scale H0 based on earliest secant information
        if hasattr(self.H0inv, "d0") and self.update_scaling:
            s0 = self.S[0]
            y0 = self.Y[0]
            d0 = s0.inner(y0) / y0.inner(y0)
            self.H0inv.d0 = d0
            self.update_scaling = False

        return theta


class BFGS:
    """
    Implement BFGS technique with backtracking inexact line search and damped updating
    See Nocedal & Wright (06), $6.2, $7.3, $18.3

    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, da) less than tolerance",      #3
                           "Cost function becomes negative"            #4
                           ]

    def __init__(self, cost, parameters=BFGS_ParameterList()):
        """
        Initialize the BFGS solver.
        Type BFGS_ParameterList().showMe() for default parameters and their description
        """
        self.cost = cost
        self.mpi_rank = self.cost.mpi_rank
        self.parameters = parameters
        self.it = 0
        self.converged = False
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0

        self.Iteration = []
        self.costValue = []
        self.costGrad = []
        self.ItLS = []

        self.BFGSop = BFGS_operator(self.parameters["BFGS_op"])

    def solve(self, z, H0inv=None, bounds_xPARAM=None):
        """
        Solve the constrained optimization problem with initial guess z
        H0inv: the initial approximated inverse of the Hessian for the BFGS operator. It has an optional method "update(z)"
               that will update the operator based on z
        bounds_xPARAM: Bound constraint (list with two entries: min and max). Can be either a scalar value or a dolfin.Vector
        """

        if bounds_xPARAM is not None:
            if hasattr(bounds_xPARAM[0], "get_local"):
                param_min = bounds_xPARAM[0].get_local()    #Assume it is a dolfin vector
            else:
                param_min = bounds_xPARAM[0]*np.ones_like(z.get_local()) #Assume it is a scalar
            if hasattr(bounds_xPARAM[1], "get_local"):
                param_max = bounds_xPARAM[1].get_local()    #Assume it is a dolfin vector
            else:
                param_max = bounds_xPARAM[1]*np.ones_like(z.get_local()) #Assume it is a scalar

        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        ls_list = self.parameters[self.parameters["globalization"]]
        c_armijo = ls_list["c_armijo"]
        max_backtracking_iter = ls_list["max_backtracking_iter"]
        print_level = self.parameters["print_level"]

        self.BFGSop.parameters["BFGS_damping"] = self.parameters["BFGS_op"]["BFGS_damping"]
        self.BFGSop.parameters["memory_limit"] = self.parameters["BFGS_op"]["memory_limit"]
        self.BFGSop.set_H0inv(H0inv)

        self.it = 0
        self.Iteration.append(self.it)
        self.converged = False
        self.ncalls += 1

        zhat = self.cost.generate_optimization()
        dz = self.cost.generate_optimization()
        z_star = self.cost.generate_optimization()

        cost_old = self.cost.costValue(z)
        self.costValue.append(cost_old)

        if self.mpi_rank == 0:
            if(print_level >= 0):
                print("\n {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "It", "cost", "(g,dm)", "||g||L2", "alpha", "theta"))
                print( "{:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                self.it, cost_old, np.inf, np.inf, 1.0, 1.0))
            f = open("iterate.dat","a+")
            f.write("\n {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "It", "cost", "(g,dm)", "||g||L2", "alpha", "theta"))
            f.write("\n\n {:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                self.it, cost_old, np.inf, np.inf, 1.0, 1.0))

        while (self.it < max_iter) and (self.converged == False):

            if hasattr(self.BFGSop.H0inv, "setPoint"):
                self.BFGSop.H0inv.setPoint(z)

            dz_old = dz.copy()

            # # check gradient
            # if self.it > 0 and np.mod(self.it, 5) == 0:
            #     self.cost.checkGradient(z, self.cost.dlcomm, plotFlag =False)

            dz, gradnorm = self.cost.costGradient(z)
            self.costGrad.append(gradnorm)
            # Update BFGS
            if self.it > 0:
                s = zhat * alpha
                y = dz - dz_old
                theta = self.BFGSop.update(s, y)
            else:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                theta = 1.0

            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break

            self.it += 1
            self.Iteration.append(self.it)

            # compute search direction with BFGS:
            self.BFGSop.solve(zhat, -dz)

            # backtracking line-search
            alpha = 1.0
            descent = 0
            n_backtrack = 0
            dz_zhat = dz.inner(zhat)
            while descent == 0 and n_backtrack < max_backtracking_iter:
                z_star.zero()
                z_star.axpy(1., z)
                z_star.axpy(alpha, zhat)
                if bounds_xPARAM is not None:
                    z_star.set_local(np.maximum(z_star.get_local(), param_min))
                    z_star.set_local(np.minimum(z_star.get_local(), param_max))
                    z_star.apply("")

                cost_new = self.cost.costValue(z_star)

                # Check if armijo conditions are satisfied
                if cost_new > 0 and ((cost_new < cost_old + alpha * c_armijo * dz_zhat) or (-dz_zhat <= self.parameters["gdm_tolerance"])):
                    cost_old = cost_new
                    self.costValue.append(cost_old)
                    descent = 1
                    z.zero()
                    z.axpy(1., z_star)
                else:
                    n_backtrack += 1
                    alpha *= 0.5

            self.ItLS.append(n_backtrack)

            if self.mpi_rank == 0:
                if (print_level >= 0):
                    print("\n {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "It", "cost", "(g,dm)", "||g||L2", "alpha", "theta"))
                    print( "{:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                    self.it, cost_new, dz_zhat, gradnorm, alpha, theta))
                f.write(" \n {:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                self.it, cost_new, dz_zhat, gradnorm, alpha, theta))

            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break

            if -dz_zhat <= self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break

            # if cost_new < 0:
            #     self.converged = True
            #     self.reason = 4
            #     break

        self.final_grad_norm = gradnorm
        self.final_cost      = cost_new

        if self.mpi_rank == 0:
            print(self.termination_reasons[self.reason])
            f.write("\n\n" + "Termination reason: "+ self.termination_reasons[self.reason] + "\n\n\n")
            f.close()

        result = dict()
        result["Iteration"] = self.Iteration
        result["costValue"] = self.costValue
        result["costGrad"] = self.costGrad
        result["ItLS"] = self.ItLS

        return result
