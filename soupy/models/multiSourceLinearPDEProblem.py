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

import dolfin as dl
# import sys
# sys.path.append('../')
from ..utils.variables import STATE, PARAMETER, ADJOINT
from ..utils.linalg import Transpose
from ..utils.vector2function import vector2Function
from ..utils.checkDolfinVersion import dlversion
from .PDEProblem import PDEProblem
from .blockVector import BlockVector

class MultiSourceLinearPDEProblem(PDEProblem):
    """
    Implements a multisource linear PDE problem. The user provides a 
    :code:`BlockVector` of RHS sources and this class makes uses of the fact the 
    operators for the forward, adjoint, incremental forward, and incremental 
    adjoint equations are identical across all sources. If the variational 
    form handler had a RHS the sources will be added to the RHS term. 
    
    .. note:: Multisource non-linear problems can not take advantage of this\
    class because the LHS operators are (generally) different for each PDE.
    """
    def __init__(self, Vh, varf_handler, sources, bc=[], bc0=[]):
        self.Vh = Vh
        self.varf_handler = varf_handler

        self.n_sources = sources.nv
        self.sources = sources 

        if type(bc) is dl.DirichletBC:
            self.bc = [bc]
        else:
            self.bc = bc
        if type(bc0) is dl.DirichletBC:
            self.bc0 = [bc0]
        else:
            self.bc0 = bc0
        
        [bc.apply(self.sources.data[s]) for s in range(self.n_sources) for bc in self.bc0]

        self.A  = None
        self.At = None
        self.C  = [None for i in range(self.n_sources)]
        self.Wmu = [None for i in range(self.n_sources)]
        self.Wmm = [None for i in range(self.n_sources)]
        self.Wuu = [None for i in range(self.n_sources)]
        
        self.solver         = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None
        
        self.is_fwd_linear = True

    def generate_state(self): 
        """ 
        Return a vector in the shape of the multisource state 
        """
        return BlockVector(self.Vh[STATE], self.n_sources)
    
    def generate_parameter(self):
        """ 
        Return a vector in the shape of the parameter 
        """
        return dl.Function(self.Vh[PARAMETER]).vector()
    
    def solveFwd(self, state, x, tol):
        """ 
        Solve the linear forward problem:
        
        Given :math:`m`, find :math:`u` such that
        
        .. math::
            (A(m)u,\hat{p}) = (f,\hat{p}), \quad \\forall \hat{p}

        .. note:: :code:`x[STATE]` and :code:`x[ADJOINT]` are required to be \
        :code:`BlockVectors`

        """
        if self.solver is None:
            self.solver = self._createLUSolver()
        u = dl.TrialFunction(self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.TestFunction(self.Vh[ADJOINT])
        res_form = self.varf_handler(u,m,p)
        A_form = dl.lhs(res_form)
        b_form = dl.rhs(res_form)
        #If the b_form is empty, create a '0' RHS.
        if b_form.empty():
            b_form = dl.inner(dl.Function(self.Vh[STATE]), p)*dl.dx
        A, b = dl.assemble_system(A_form, b_form, bcs=self.bc)
        self.solver.set_operator(A)
        for s in range(self.n_sources):
            self.solver.solve(state.data[s], b + self.sources.data[s])

    def solveAdj(self, adj, x, adj_rhs, tol): 
        """ 
        Solve the linear adjoint problem: 
        
        Given :math:`m, u` find :math:`p` such that
        
        .. math::
            (A^T(m)p,\hat{u}) = (f,\hat{u}), \quad \\forall \hat{u}
        
        .. note:: :code:`adj` and :code:`adj_rhs` are required to be :code:`BlockVectors`

        """
        if self.solver is None:
            self.solver = self._createLUSolver()
        u = dl.TestFunction(self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.TrialFunction(self.Vh[ADJOINT])
        res_form = self.varf_handler(u,m,p)
        A_form = dl.lhs(res_form)
        dummy_b_form = dl.inner(dl.Function(self.Vh[STATE]), u)*dl.dx
        Aadj, dummy = dl.assemble_system(A_form, dummy_b_form, bcs=self.bc0)
        self.solver.set_operator(Aadj)
        for s in range(self.n_sources):
            self.solver.solve(adj.data[s], adj_rhs.data[s])

    def evalGradientParameter(self, x, out): 

        """ 
        Given :math:`u, m, p` evaluate :math:`(\delta_m(A(m)u,p); \hat{m}),\: \\forall \hat{m}`. 

        .. note:: :code:`x[STATE]`, :code:`x[ADJOINT]` are :code:`BlockVectors`, \
        :code:`out` is :code:`dolfin.Vector`

        """
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        dm = dl.TestFunction(self.Vh[PARAMETER])
        tmp = self.generate_parameter()
        out.zero()
        for s in range(self.n_sources):
            u = vector2Function(x[STATE].data[s], self.Vh[STATE])
            p = vector2Function(x[ADJOINT].data[s], self.Vh[ADJOINT])
            res_form = self.varf_handler(u,m,p)
            tmp.zero()
            dl.assemble( dl.derivative(res_form, m, dm), tensor=tmp)
            out.axpy(1.,tmp)
            
    def setLinearizationPoint(self,x, gauss_newton_approx):
        """ 
        Set the values of the state and parameter
        for the incremental forward and adjoint solvers 

        .. note:: :code:`x` is required to be :code:`BlockVector`
        """
        self.gauss_newton_approx = gauss_newton_approx
        self.C   = [None for s in range(self.n_sources)]
        self.Wmu = [None for s in range(self.n_sources)]
        self.Wmm = [None for s in range(self.n_sources)]
        self.Wuu = [None for s in range(self.n_sources)]

        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        u = vector2Function(x[STATE].data[0], self.Vh[STATE])
        p = vector2Function(x[ADJOINT].data[0], self.Vh[ADJOINT])
        x_fun = [u,m,p]
        f_form = self.varf_handler(*x_fun)
        
        g_form = [None, None, None]
        for j in range(3):
            g_form[j] = dl.derivative(f_form, x_fun[j])
            
        self.A,  dummy = dl.assemble_system(dl.derivative(g_form[ADJOINT], u), g_form[ADJOINT], self.bc)
        self.At, dummy = dl.assemble_system(dl.derivative(g_form[STATE], p), g_form[STATE], self.bc0)

        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()
        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)

        for s in range(self.n_sources):
            u = vector2Function(x[STATE].data[s], self.Vh[STATE])
            p = vector2Function(x[ADJOINT].data[s], self.Vh[ADJOINT])
            x_fun = [u,m,p]
            
            f_form = self.varf_handler(*x_fun)
            g_form = [None,None,None]
            for j in range(3):
                g_form[j] = dl.derivative(f_form, x_fun[j])
                
            self.C[s]=dl.assemble(dl.derivative(g_form[ADJOINT],m))
            [bc.zero(self.C[s]) for bc in self.bc0]

            if not self.gauss_newton_approx:
                self.Wuu[s]=dl.assemble(dl.derivative(g_form[STATE],u))
                self.Wmu[s]=dl.assemble(dl.derivative(g_form[PARAMETER],u))
                self.Wmm[s]=dl.assemble(dl.derivative(g_form[PARAMETER],m))
                
                [bc.zero(self.Wuu[s]) for bc in self.bc0]
                Wuu_t = Transpose(self.Wuu[s])
                [bc.zero(Wuu_t) for bc in self.bc0]
                self.Wuu[s] = Transpose(Wuu_t)

                self.Wmu[s] = dl.assemble(dl.derivative(g_form[PARAMETER],u))
                Wmu_t = Transpose(self.Wmu[s])
                [bc.zero(Wmu_t) for bc in self.bc0]
                self.Wmu[s] = Transpose(Wmu_t)

    def solveIncremental(self, out, rhs, is_adj, mytol):
        """ 
        If :code:`is_adj = False`, solve the forward incremental system:
        
        Given :math:`u, m`, find :math:`\\tilde{u}` s.t.:
        
        .. math::
            \delta_{pu} F(u,m,p; \hat{p}, \\tilde{u}) = \mbox{rhs}, \quad \\forall \hat{p}.
            
        If :code:`is_adj = True`, solve the adj incremental system:
        
        Given :math:`u, m`, find :math:`\\tilde{p}` s.t.:
        
        .. math::
            \delta_{up} F(u,m,p; \hat{u}, \\tilde{p}) = \mbox{rhs}, \quad \\forall \hat{u}.

        .. note:: :code:`out` and :code:`rhs` are required to be :code:`BlockVectors`
        """
        if is_adj:
            for s in range(self.n_sources):
                self.solver_adj_inc.solve(out.data[s], rhs.data[s])
        else:
            for s in range(self.n_sources):
                self.solver_fwd_inc.solve(out.data[s], rhs.data[s])

    def _apply_ij_newton(self, i, j, dir, out):  
        out.zero()
        
        if i == PARAMETER:
            tmp = self.generate_parameter()
            
        for k in range(self.n_sources):
            if i == STATE:
                assert (type(out) is BlockVector),"type(out) is NOT BlockVector! apply_ij has been incorrectly called"
                if j == STATE:
                    self.Wuu[k].mult(dir.data[k], out.data[k])
                elif j == PARAMETER:
                    self.Wmu[k].transpmult(dir, out.data[k])
                elif j == ADJOINT:
                    self.A.transpmult(dir.data[k], out.data[k])
                else:
                    raise IndexError("Invalid index j for apply_ij!  apply_ij has been incorrectly called")
            elif i == PARAMETER:
                if j == STATE:
                    tmp.zero()
                    self.Wmu[k].mult(dir.data[k], tmp) 
                    out.axpy(1., tmp)
                elif j == PARAMETER:
                    tmp.zero()
                    self.Wmm[k].mult(dir, tmp)
                    out.axpy(1., tmp)
                elif j == ADJOINT:
                    tmp.zero()
                    self.C[k].transpmult(dir.data[k], tmp)
                    out.axpy(1., tmp)
                else:
                    raise IndexError("Invalid index j for apply_ij!  apply_ij has been incorrectly called")
            elif i == ADJOINT:
                assert (type(out) is BlockVector),"type(out) is NOT BlockVector! apply_ij has been incorrectly called"
                if j == STATE:
                    self.A.mult(dir.data[k], out.data[k])
                elif j == PARAMETER:
                    self.C[k].mult(dir, out.data[k])
                else:
                    raise IndexError("Invalid index j for apply_ij!  apply_ij has been incorrectly called")
            else:
                 raise IndexError("Invalid index i for apply_ij!  apply_ij has been incorrectly called")

    def _apply_ij_gauss_newton(self, i, j, dir, out): 
        
        if i not in [STATE,PARAMETER,ADJOINT]:
            raise IndexError("Invalid index i for apply_ij!  apply_ij has been incorrectly called")
        
        if j not in [STATE,PARAMETER,ADJOINT]:
            raise IndexError("Invalid index j for apply_ij!  apply_ij has been incorrectly called")
        
        out.zero()
        
        if i == PARAMETER:
            tmp = self.generate_parameter()
            
        for k in range(self.n_sources):
            if i == STATE and j == ADJOINT:
                self.A.transpmult(dir.data[k], out.data[k])
            elif i == PARAMETER and j == ADJOINT:
                tmp.zero()
                self.C[k].transpmult(dir.data[k], tmp)
                out.axpy(1., tmp)
            elif i == ADJOINT and j == STATE:
                self.A.mult(dir.data[k], out.data[k])
            elif i == ADJOINT and j == PARAMETER:
                self.C[k].mult(dir, out.data[k])
            else:
                return

    def apply_ij(self,i,j,dir, out): 
        """

        .. note:: Since :code:`setLinearizationPoint()` should ALWAYS be called before :code:`apply_ij()`, \
        it is safe to assume :code:`self.gauss_newton_approx` exists 

        """
        if self.gauss_newton_approx:
            self._apply_ij_gauss_newton(i,j,dir, out)
        else:
            self._apply_ij_newton(i,j,dir, out)

    def _createLUSolver(self):
        if dlversion() <= (1,6,0):
            return dl.PETScLUSolver()
        else:
            return dl.PETScLUSolver(self.Vh[STATE].mesh().mpi_comm())
                            