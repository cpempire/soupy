from __future__ import absolute_import, division, print_function

# import sys
# sys.path.append('../')

from ..utils.variables import STATE, PARAMETER, ADJOINT


class ReducedHessianSVD:

    def __init__(self, pde, qoi, tol):
        self.pde = pde
        self.qoi = qoi
        self.rhs_fwd = pde.generate_state()
        self.rhs_adj = pde.generate_state()
        self.rhs_adj2 = pde.generate_state()
        self.rhs_adj3 = pde.generate_state()
        self.rhs_adj4 = pde.generate_state()
        self.mhelp = pde.generate_parameter()
        self.Hmhat1 = pde.generate_parameter()
        self.tol = tol

    # implement init_vector for Hessian in doublePassG
    def init_vector(self, m, dim):
        self.pde.init_parameter(m)

    # implement mult for Hessian in doublePassG
    def mult(self, mhat1, Hmat1):
        xhat = self.pde.generate_state()
        yhat = self.pde.generate_state()
        self.pde.apply_ij(ADJOINT,PARAMETER, mhat1, self.rhs_fwd)
        self.pde.solveIncremental(xhat, -self.rhs_fwd, False, self.tol) # False for forward,
        self.pde.apply_ij(STATE, STATE, xhat, self.rhs_adj)
        self.pde.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj2)
        self.rhs_adj.axpy(1., self.rhs_adj2)
        self.qoi.apply_ij(STATE,STATE,xhat,self.rhs_adj3)
        self.rhs_adj.axpy(1., self.rhs_adj3)
        self.qoi.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj4)
        self.rhs_adj.axpy(1., self.rhs_adj4)
        self.pde.solveIncremental(yhat, -self.rhs_adj, True, self.tol) # True for adjoint,
        self.pde.apply_ij(PARAMETER,PARAMETER, mhat1, self.Hmhat1)
        self.pde.apply_ij(PARAMETER,ADJOINT, yhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)
        self.pde.apply_ij(PARAMETER,STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)
        self.qoi.apply_ij(PARAMETER, ADJOINT, yhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)
        self.qoi.apply_ij(PARAMETER, STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)

        Hmat1[:] = self.Hmhat1[:]

    def HessianInner(self, mhat1, mhat2):
        xhat = self.pde.generate_state()
        yhat = self.pde.generate_state()
        self.pde.apply_ij(ADJOINT,PARAMETER, mhat1, self.rhs_fwd)
        self.pde.solveIncremental(xhat, -self.rhs_fwd, False, self.tol) # False for forward,
        self.pde.apply_ij(STATE, STATE, xhat, self.rhs_adj)
        self.pde.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj2)
        self.rhs_adj.axpy(1., self.rhs_adj2)
        self.qoi.apply_ij(STATE,STATE,xhat,self.rhs_adj3)
        self.rhs_adj.axpy(1., self.rhs_adj3)
        self.qoi.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj4)
        self.rhs_adj.axpy(1., self.rhs_adj4)
        self.pde.solveIncremental(yhat, -self.rhs_adj, True, self.tol) # True for adjoint,
        self.pde.apply_ij(PARAMETER,PARAMETER, mhat1, self.Hmhat1)
        self.pde.apply_ij(PARAMETER,ADJOINT,yhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)
        self.pde.apply_ij(PARAMETER,STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)
        self.qoi.apply_ij(PARAMETER, ADJOINT, yhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)
        self.qoi.apply_ij(PARAMETER, STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)

        return mhat2.inner(self.Hmhat1), xhat, yhat
