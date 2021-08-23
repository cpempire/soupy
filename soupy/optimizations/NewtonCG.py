# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

from __future__ import absolute_import, division, print_function

import math
import numpy as np

# import sys
# sys.path.append('../')
from ..models.reducedHessian import ReducedHessian
from ..utils.parameterList import ParameterList
from ..utils.variables import STATE, PARAMETER, ADJOINT
from .cgsolverSteihaug import CGSolverSteihaug


def LS_ParameterList():
    """
    Generate a ParameterList for line search globalization.
    type: :code:`LS_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["c_armijo"]              = [1e-4, "Armijo constant for sufficient reduction"]
    parameters["max_backtracking_iter"] = [20, "Maximum number of backtracking iterations"]
    
    return ParameterList(parameters)

def TR_ParameterList():
    """
    Generate a ParameterList for Trust Region globalization.
    type: :code:`RT_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["eta"] = [0.05, "Reject step if (actual reduction)/(predicted reduction) < eta"]
    
    return ParameterList(parameters)

def ReducedSpaceNewtonCG_ParameterList():
    """
    Generate a ParameterList for ReducedSpaceNewtonCG.
    type: :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdm_tolerance"]         = [1e-18, "we converge when (g,dm) <= gdm_tolerance"]
    parameters["max_iter"]              = [20, "maximum number of iterations"]
    parameters["inner_rel_tolerance"]   = [1e-9, "relative tolerance used for the solution of the forward, adjoint, and incremental (fwd,adj) problems"]
    parameters["globalization"]         = ["LS", "Globalization technique: line search (LS)  or trust region (TR)"]
    parameters["print_level"]           = [0, "Control verbosity of printing screen"]
    parameters["GN_iter"]               = [5, "Number of Gauss Newton iterations before switching to Newton"]
    parameters["cg_coarse_tolerance"]   = [.5, "Coarsest tolerance for the CG method (Eisenstat-Walker)"]
    parameters["LS"]                    = [LS_ParameterList(), "Sublist containing LS globalization parameters"]
    parameters["TR"]                    = [TR_ParameterList(), "Sublist containing TR globalization parameters"]
    
    return ParameterList(parameters)
  
    

class ReducedSpaceNewtonCG:
    
    """
    Inexact Newton-CG method to solve constrained optimization problems in the reduced optimization space.
    The Newton system is solved inexactly by early termination of CG iterations via Eisenstat-Walker
    (to prevent oversolving) and Steihaug (to avoid negative curvature) criteria.
    Globalization is performed using one of the following methods:

    - line search (LS) based on the armijo sufficient reduction condition; or
    - trust region (TR) based on the prior preconditioned norm of the update direction.

    The stopping criterion is based on a control on the norm of the gradient and a control of the
    inner product between the gradient and the Newton direction.
       
    """
    termination_reasons = [
                           "Maximum number of Iteration reached",      #0
                           "Norm of the gradient less than tolerance", #1
                           "Maximum number of backtracking reached",   #2
                           "Norm of (g, dm) less than tolerance"       #3
                           ]
    
    def __init__(self, cost, parameters=ReducedSpaceNewtonCG_ParameterList()):
        """
        Initialize the ReducedSpaceNewtonCG.
        Type :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for list of default parameters
        and their descriptions.
        """
        self.cost = cost
        self.mpi_rank = self.cost.mpi_rank
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

    def solve(self, z):
        """

        Input: 
            :code:`z` represents the initial guess.
            :code:`z` will be overwritten on return.
        """

        if z is None:
            z = self.cost.generate_optimization()

        if self.parameters["globalization"] == "LS":
            return self._solve_ls(z)
        elif self.parameters["globalization"] == "TR":
            return self._solve_tr(z)
        else:
            raise ValueError(self.parameters["globalization"])
        
    def _solve_ls(self, z):
        """
        Solve the constrained optimization problem with initial guess :code:`x`.
        """
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        innerTol = self.parameters["inner_rel_tolerance"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
        
        c_armijo = self.parameters["LS"]["c_armijo"]
        max_backtracking_iter = self.parameters["LS"]["max_backtracking_iter"]

        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        zhat = self.cost.generate_optimization()
        dz = self.cost.generate_optimization()
        z_star = self.cost.generate_optimization()

        cost_old = self.cost.costValue(z)
        self.costValue.append(cost_old)

        if self.mpi_rank == 0:
            if(print_level >= 0):
                print("\n {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "It", "cost", "(g,dm)", "||g||L2", "alpha"))
                print( "{:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                self.it, cost_old, np.inf, np.inf, 1.0))
            f = open("iterate.dat","a+")
            f.write("\n {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "It", "cost", "(g,dm)", "||g||L2", "alpha"))
            f.write("\n\n {:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                self.it, cost_old, np.inf, np.inf, 1.0))

        while (self.it < max_iter) and (self.converged is False):

            # # check Hessian
            # if self.it >= 0 and np.mod(self.it, 5) == 0:
            #     self.cost.checkHessian(z, self.cost.dlcomm, plotFlag=False)

            dz, gradnorm = self.cost.costGradient(z)
            self.costGrad.append(gradnorm)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1
            
            tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))
            
            HessApply = self.cost
            solver = CGSolverSteihaug()
            solver.set_operator(HessApply)
            solver.set_preconditioner(self.cost.penalization)
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["max_iter"] = 20
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = print_level-1
            
            solver.solve(zhat, -dz)
            self.total_cg_iter += HessApply.ncalls
            
            alpha = 1.0
            descent = 0
            n_backtrack = 0
            
            dz_zhat = dz.inner(zhat)
            while descent == 0 and n_backtrack < max_backtracking_iter:
                z_star.zero()
                z_star.axpy(1., z)
                z_star.axpy(alpha, zhat)

                cost_new = self.cost.costValue(z_star)

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

            self.ItLS.append(n_backtrack)

            if self.mpi_rank == 0:
                if (print_level >= 0):
                    print("\n {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                        "It", "cost", "(g,dm)", "||g||L2", "alpha"))
                    print("{:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                        self.it, cost_new, dz_zhat, gradnorm, alpha))
                f.write(" \n {:<10d} {:<10.2e} {:<10.2e} {:<10.2e} {:<10.2e}".format(
                    self.it, cost_new, dz_zhat, gradnorm, alpha))

            if n_backtrack == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break

            if -dz_zhat <= self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break


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
    
    def _solve_tr(self, x):
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        innerTol = self.parameters["inner_rel_tolerance"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
        
        eta_TR = self.parameters["TR"]["eta"]
        delta_TR = None
        
        
        self.model.solveFwd(x[STATE], x, innerTol)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        mhat = self.model.generate_vector(PARAMETER) 
        R_mhat = self.model.generate_vector(PARAMETER)   

        mg = self.model.generate_vector(PARAMETER)
        
        x_star = [None, None, None] + x[3::]
        x_star[STATE]     = self.model.generate_vector(STATE)
        x_star[PARAMETER] = self.model.generate_vector(PARAMETER)
        
        cost_old, reg_old, misfit_old = self.model.cost(x)
        while (self.it < max_iter) and (self.converged == False):
            self.model.solveAdj(x[ADJOINT], x, innerTol)
            
            self.model.setPointForHessianEvaluations(x, gauss_newton_approx=(self.it < GN_iter) )
            gradnorm = self.model.evalGradientParameter(x, mg)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1

            tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))
            
            HessApply = ReducedHessian(self.model, innerTol)
            solver = CGSolverSteihaug(comm = self.model.prior.R.mpi_comm())
            solver.set_operator(HessApply)
            solver.set_preconditioner(self.model.Rsolver())
            if self.it > 1:
                solver.set_TR(delta_TR, self.model.prior.R)
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["print_level"] = print_level-1
            
            solver.solve(mhat, -mg)
            self.total_cg_iter += HessApply.ncalls

            if self.it == 1:
                self.model.prior.R.mult(mhat,R_mhat)
                mhat_Rnorm = R_mhat.inner(mhat)
                delta_TR = max(math.sqrt(mhat_Rnorm),1)

            x_star[PARAMETER].zero()
            x_star[PARAMETER].axpy(1., x[PARAMETER])
            x_star[PARAMETER].axpy(1., mhat)   #m_star = m +mhat
            x_star[STATE].zero()
            x_star[STATE].axpy(1., x[STATE])      #u_star = u
            self.model.solveFwd(x_star[STATE], x_star, innerTol)
            cost_star, reg_star, misfit_star = self.model.cost(x_star)
            ACTUAL_RED = cost_old - cost_star
            #Calculate Predicted Reduction
            H_mhat = self.model.generate_vector(PARAMETER)
            H_mhat.zero()
            HessApply.mult(mhat,H_mhat)
            mg_mhat = mg.inner(mhat)
            PRED_RED = -0.5*mhat.inner(H_mhat) - mg_mhat
            # print( "PREDICTED REDUCTION", PRED_RED, "ACTUAL REDUCTION", ACTUAL_RED)
            rho_TR = ACTUAL_RED/PRED_RED


            # Nocedal and Wright Trust Region conditions (page 69)
            if rho_TR < 0.25:
                delta_TR *= 0.5
            elif rho_TR > 0.75 and solver.reasonid == 3:
                delta_TR *= 2.0
            

            # print( "rho_TR", rho_TR, "eta_TR", eta_TR, "rho_TR > eta_TR?", rho_TR > eta_TR , "\n")
            if rho_TR > eta_TR:
                x[PARAMETER].zero()
                x[PARAMETER].axpy(1.0,x_star[PARAMETER])
                x[STATE].zero()
                x[STATE].axpy(1.0,x_star[STATE])
                cost_old = cost_star
                reg_old = reg_star
                misfit_old = misfit_star
                accept_step = True
            else:
                accept_step = False
                
                            
            if(print_level >= 0) and (self.it == 1):
                print( "\n{0:3} {1:3} {2:15} {3:15} {4:15} {5:15} {6:14} {7:14} {8:14} {9:11} {10:14}".format(
                      "It", "cg_it", "cost", "misfit", "reg", "(g,dm)", "||g||L2", "TR Radius", "rho_TR", "Accept Step","tolcg") )
                
            if print_level >= 0:
                print( "{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e} {7:14e} {8:14e} {9:11} {10:14e}".format(
                        self.it, HessApply.ncalls, cost_old, misfit_old, reg_old, mg_mhat, gradnorm, delta_TR, rho_TR, accept_step,tolcg) )
                

            #TR radius can make this term arbitrarily small and prematurely exit.
            if -mg_mhat <= self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 3
                break
                            
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_old
        return x
