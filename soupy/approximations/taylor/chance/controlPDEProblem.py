'''
Created on March 27, 2018

@author: peng chen
'''

from __future__ import absolute_import, division, print_function

import dolfin as dl
dl.dx = dl.dx(metadata={'quadrature_degree':2, "representation":'uflacs'}) #, "representation":'uflacs'

# import sys
# sys.path.append('../../')

from ...utils import *
from ...models.PDEProblem import PDEProblem


class ControlPDEProblem(PDEProblem):
    """ Consider the PDE Problem with control:
        Given m, z, find u s.t.
        R(u, m, p, z) = ( r(u, m, z), p) = 0 for all p.
        Here R is linear in p, but it may be non-linear in u and m.
    """

    def __init__(self, Vh, residual, bc, bc0, is_fwd_linear=False):
        self.Vh = Vh
        self.residual = residual
        if type(bc) is dl.DirichletBC:
            self.bc = [bc]
        else:
            self.bc = bc
        if type(bc0) is dl.DirichletBC:
            self.bc0 = [bc0]
        else:
            self.bc0 = bc0

        self.A = None
        self.At = None
        self.C = None
        self.Wmu = None
        self.Wmm = None
        self.Wuu = None

        self.solver = None
        self.Asolver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None

        self.is_fwd_linear = is_fwd_linear

    def generate_all(self):
        """ return a vector in the shape of the [state,parameter,adjoint,optimization] """
        return [dl.Function(self.Vh[i]).vector() for i in range(4)]

    def generate_state(self):
        """ return a vector in the shape of the state """
        return dl.Function(self.Vh[STATE]).vector()

    def generate_parameter(self):
        """ return a vector in the shape of the parameter """
        return dl.Function(self.Vh[PARAMETER]).vector()

    def generate_optimization(self):
        """ return a vector in the shape of the control """
        return dl.Function(self.Vh[OPTIMIZATION]).vector()

    def init_parameter(self, m):
        """ initialize the parameter """
        dummy = self.generate_parameter()
        m.init( dummy.mpi_comm(), dummy.local_range() )

    def init_optimization(self, z):
        """ initialize the parameter """
        dummy = self.generate_optimization()
        z.init( dummy.mpi_comm(), dummy.local_range() )

    def solveFwd(self, state, x, tol):
        """ Solve the possibly nonlinear Fwd Problem:
        Given m and z, find u such that
        \delta_p R(u,m,p,z;\hat_p) = 0 \for all \hat_p
        """
        if self.solver is None:
            self.solver = self._createLUSolver()
        if self.is_fwd_linear:
            u = dl.TrialFunction(self.Vh[STATE])
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])
            z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])
            res_form = self.residual(u, m, p, z)
            A_form = dl.lhs(res_form)
            b_form = dl.rhs(res_form)
            A, b = dl.assemble_system(A_form, b_form, bcs=self.bc)
            self.solver.set_operator(A)
            self.solver.solve(state, b)
        else:
            u = vector2Function(x[STATE], self.Vh[STATE])
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = dl.TestFunction(self.Vh[ADJOINT])
            z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])
            res_form = self.residual(u, m, p, z)
            dl.solve(res_form == 0, u, self.bc)
            state.zero()
            state.axpy(1., u.vector())

    def solveAdj(self, adj, x, adj_rhs, tol):
        """ Solve the linear Adj Problem:
            Given m, u, z; find p such that
            \delta_u F(u,m,p, z;\hat_u) = 0 \for all \hat_u
        """
        if self.solver is None:
            self.solver = self._createLUSolver()

        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.Function(self.Vh[ADJOINT])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])
        du = dl.TestFunction(self.Vh[STATE])
        dp = dl.TrialFunction(self.Vh[ADJOINT])
        res_form = self.residual(u, m, p, z)
        adj_form = dl.derivative( dl.derivative(res_form, u, du), p, dp )
        Aadj, dummy = dl.assemble_system(adj_form, dl.inner(u,du)*dl.dx, self.bc0)
        self.solver.set_operator(Aadj)
        self.solver.solve(adj, adj_rhs)

    def evalGradientParameter(self, x, out):
        """Given u, m, p; eval \delta_m F(u, m, p; \hat_m) \for all \hat_m """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])
        dm = dl.TestFunction(self.Vh[PARAMETER])
        res_form = self.residual(u, m, p, z)
        out.zero()
        dl.assemble( dl.derivative(res_form, m, dm), tensor=out)

    def evalGradientControl(self, x, out):
        """Given u, m, p; eval \delta_m F(u, m, p; \hat_m) \for all \hat_m """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])
        dz = dl.TestFunction(self.Vh[OPTIMIZATION])
        res_form = self.residual(u, m, p, z)
        out.zero()
        dl.assemble( dl.derivative(res_form, z, dz), tensor=out)

    def setLinearizationPoint(self, x, gauss_newton_approx = False):
        """ Set the values of the state and parameter
            for the incremental Fwd and Adj solvers """
        x_fun = [vector2Function(x[i], self.Vh[i]) for i in range(4)]

        f_form = self.residual(*x_fun)

        g_form = [None,None,None,None]
        for i in range(4):
            g_form[i] = dl.derivative(f_form, x_fun[i])

        self.A, dummy = dl.assemble_system(dl.derivative(g_form[ADJOINT],x_fun[STATE]), g_form[ADJOINT], self.bc0)
        self.At, dummy = dl.assemble_system(dl.derivative(g_form[STATE],x_fun[ADJOINT]),  g_form[STATE], self.bc0)
        self.C = dl.assemble(dl.derivative(g_form[ADJOINT],x_fun[PARAMETER]))
        [bc.zero(self.C) for bc in self.bc0]

        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()

        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)

        if self.Asolver is None:
            self.Asolver = self._createLUSolver()

        self.Asolver.set_operator(self.A)

        if gauss_newton_approx:
            self.Wuu = None
            self.Wmu = None
            self.Wmm = None
        else:
            self.Wuu = dl.assemble(dl.derivative(g_form[STATE],x_fun[STATE]))
            [bc.zero(self.Wuu) for bc in self.bc0]
            Wuu_t = Transpose(self.Wuu)
            [bc.zero(Wuu_t) for bc in self.bc0]
            self.Wuu = Transpose(Wuu_t)
            self.Wmu = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[STATE]))
            Wmu_t = Transpose(self.Wmu)
            [bc.zero(Wmu_t) for bc in self.bc0]
            self.Wmu = Transpose(Wmu_t)
            self.Wmm = dl.assemble(dl.derivative(g_form[PARAMETER],x_fun[PARAMETER]))

    def solveIncremental(self, out, rhs, is_adj, mytol):
        """ If is_adj = False:
            Solve the forward incremental system:
            Given u, m, p, z, find \tilde_u s.t.:
            \delta_{pu} R(u, m, p, z; \hat_p, \tilde_u) = rhs for all \hat_p.

            If is_adj = True:
            Solve the adj incremental system:
            Given u, m, p, z, find \tilde_p s.t.:
            \delta_{up} R(u, m, p, z; \hat_u, \tilde_p) = rhs for all \delta_u.
        """
        if is_adj:
            self.solver_adj_inc.solve(out, rhs)
        else:
            self.solver_fwd_inc.solve(out, rhs)

    def apply_ij(self,i,j, dir, out):
        """
            Given u, m, p, z; compute
            \delta_{ij} R(u, m, p, z; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """
        KKT = {}
        KKT[STATE,STATE] = self.Wuu
        KKT[PARAMETER, STATE] = self.Wmu
        KKT[PARAMETER, PARAMETER] = self.Wmm
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C

        if i >= j:
            if KKT[i,j] is None:
                out.zero()
            else:
                KKT[i,j].mult(dir, out)
        else:
            if KKT[j,i] is None:
                out.zero()
            else:
                KKT[j,i].transpmult(dir, out)

    def _createLUSolver(self):
        if dlversion() <= (1,6,0):
            return dl.PETScLUSolver()
        else:
            return dl.PETScLUSolver(self.Vh[STATE].mesh().mpi_comm() )

##############################################################


    def forSolveAdjIncrementalAdj(self, x, mhat):
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])
        mhat_fun = vector2Function(mhat, self.Vh[PARAMETER])

        f_form = self.residual(u, m, p, z)
        p_test = dl.TestFunction(self.Vh[ADJOINT])

        dmr = dl.derivative(f_form, m, mhat_fun)
        dmyr = dl.derivative(dmr, p, p_test)
        dmyr = dl.assemble(dmyr)
        [bc.apply(dmyr) for bc in self.bc0]

        return dmyr

    def forSolveAdjIncrementalFwd(self, x, mhat, uhatstar, qoi):
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])
        mhat_fun = vector2Function(mhat, self.Vh[PARAMETER])
        uhatstar_fun = vector2Function(uhatstar, self.Vh[STATE])

        f_form = self.residual(u, m, p, z)
        u_test = dl.TestFunction(self.Vh[STATE])

        dmr = dl.derivative(f_form, m, mhat_fun)
        dmxr = dl.derivative(dmr, u, u_test)
        dmxr = dl.assemble(dmxr)
        [bc.apply(dmxr) for bc in self.bc0]

        dxr = dl.derivative(f_form, u, uhatstar_fun)
        dxxr = dl.derivative(dxr, u, u_test)
        dxxr = dl.assemble(dxxr)
        [bc.apply(dxxr) for bc in self.bc0]

        dxxq = self.generate_state()
        qoi.apply_ij(STATE, STATE, uhatstar, dxxq)
        [bc.apply(dxxq) for bc in self.bc0]

        return dmxr, dxxr, dxxq

    def forSolveAdjAdj(self, x, uhat, uhatstar, mhat, mhatstar):
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])
        mhat_fun = vector2Function(mhat, self.Vh[PARAMETER])
        mhatstar_fun = vector2Function(mhatstar, self.Vh[PARAMETER])
        uhat_fun = vector2Function(uhat, self.Vh[STATE])
        uhatstar_fun = vector2Function(uhatstar, self.Vh[STATE])

        f_form = self.residual(u, m, p, z)
        p_test = dl.TestFunction(self.Vh[ADJOINT])

        dxr = dl.derivative(f_form, u, uhatstar_fun)

        dxxr = dl.derivative(dxr, u, uhat_fun)
        dxxyr = dl.derivative(dxxr, p, p_test)
        dxxyr = dl.assemble(dxxyr)
        #import pdb
        #pdb.set_trace()
        [bc.apply(dxxyr) for bc in self.bc0]

        dxmr = dl.derivative(dxr,m,mhat_fun)
        dxmyr = dl.derivative(dxmr, p, p_test)
        dxmyr = dl.assemble(dxmyr)
        [bc.apply(dxmyr) for bc in self.bc0]

        dmr = dl.derivative(f_form, m, mhat_fun)

        dmyr = dl.derivative(dmr, p, p_test)
        dmyr = dl.assemble(dmyr)
        [bc.apply(dmyr) for bc in self.bc0]

        dmr = dl.derivative(f_form, m, mhatstar_fun)

        dmmr = dl.derivative(dmr, m, mhat_fun)
        dmmyr = dl.derivative(dmmr, p, p_test)
        dmmyr = dl.assemble(dmmyr)
        [bc.apply(dmmyr) for bc in self.bc0]


        dmxr = dl.derivative(dmr, u, uhat_fun)
        dmxyr = dl.derivative(dmxr, p, p_test)
        dmxyr = dl.assemble(dmxyr)
        [bc.apply(dmxyr) for bc in self.bc0]

        return dmyr, dxxyr, dxmyr, dmmyr, dmxyr

    def forSolveAdjFwd(self, x, uhat, uhatstar, mhat, mhatstar, phat, phatstar, qoi):
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])
        mhat_fun = vector2Function(mhat, self.Vh[PARAMETER])
        mhatstar_fun = vector2Function(mhatstar, self.Vh[PARAMETER])
        uhat_fun = vector2Function(uhat, self.Vh[STATE])
        phat_fun = vector2Function(phat, self.Vh[ADJOINT])
        uhatstar_fun = vector2Function(uhatstar, self.Vh[STATE])
        phatstar_fun = vector2Function(phatstar, self.Vh[ADJOINT])

        f_form = self.residual(u, m, p, z)
        u_test = dl.TestFunction(self.Vh[STATE])

        dyr = dl.derivative(f_form, p, phatstar_fun)

        dyxr = dl.derivative(dyr, u, uhat_fun)
        dyxxr = dl.derivative(dyxr, u, u_test)
        dyxxr = dl.assemble(dyxxr)
        [bc.apply(dyxxr) for bc in self.bc0]

        dymr = dl.derivative(dyr, m, mhat_fun)
        dymxr = dl.derivative(dymr, u, u_test)
        dymxr = dl.assemble(dymxr)
        [bc.apply(dymxr) for bc in self.bc0]

        dxr = dl.derivative(f_form, u, uhatstar_fun)

        dxyr = dl.derivative(dxr, p, phat_fun)
        dxyxr = dl.derivative(dxyr, u, u_test)
        dxyxr = dl.assemble(dxyxr)
        [bc.apply(dxyxr) for bc in self.bc0]

        dxxr = dl.derivative(dxr, u, uhat_fun)
        dxxxr = dl.derivative(dxxr, u, u_test)
        dxxxr = dl.assemble(dxxxr)
        [bc.apply(dxxxr) for bc in self.bc0]

        dxmr = dl.derivative(dxr, m, mhat_fun)
        dxmxr = dl.derivative(dxmr, u, u_test)
        dxmxr = dl.assemble(dxmxr)
        [bc.apply(dxmxr) for bc in self.bc0]

        dxxxq = self.generate_state()
        qoi.apply_ijk(STATE, STATE, STATE, uhatstar, uhat, dxxxq)
        [bc.apply(dxxxq) for bc in self.bc0]

        dmr = dl.derivative(f_form, m, mhat_fun)

        dmxr = dl.derivative(dmr, u, u_test)
        dmxr_ass = dl.assemble(dmxr)
        [bc.apply(dmxr_ass) for bc in self.bc0]

        dmr = dl.derivative(f_form, m, mhatstar_fun)

        dmmr = dl.derivative(dmr, m, mhat_fun)
        dmmxr = dl.derivative(dmmr, u, u_test)
        dmmxr = dl.assemble(dmmxr)
        [bc.apply(dmmxr) for bc in self.bc0]

        dmyr = dl.derivative(dmr,p,phat_fun)
        dmyxr = dl.derivative(dmyr, u, u_test)
        dmyxr = dl.assemble(dmyxr)
        [bc.apply(dmyxr) for bc in self.bc0]

        dmxr = dl.derivative(dmr, u, uhat_fun)
        dmxxr = dl.derivative(dmxr, u, u_test)
        dmxxr = dl.assemble(dmxxr)
        [bc.apply(dmxxr) for bc in self.bc0]

        dxmxq = self.generate_state()
        qoi.apply_ijk(STATE,PARAMETER,STATE,uhatstar,mhat,dxmxq)
        [bc.apply(dxmxq) for bc in self.bc0]

        dmmxq = self.generate_state()
        qoi.apply_ijk(PARAMETER,PARAMETER,STATE,mhatstar,mhat,dmmxq)
        [bc.apply(dmmxq) for bc in self.bc0]

        dmxxq = self.generate_state()
        qoi.apply_ijk(PARAMETER,STATE,STATE,mhatstar,uhat,dmxxq)
        [bc.apply(dmxxq) for bc in self.bc0]

        return dmxr_ass, dyxxr, dymxr, dxyxr, dxxxr, dxmxr, dxxxq, dmmxr, dmyxr, dmxxr, dxmxq, dmmxq, dmxxq

    def gradientControl(self, x, ustar, pstar, uhat, uhatstar, mhat, mhatstar, phat, phatstar, qoi):
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])
        ustar_fun = vector2Function(ustar, self.Vh[STATE])
        pstar_fun = vector2Function(pstar, self.Vh[ADJOINT])
        uhat_fun = vector2Function(uhat, self.Vh[STATE])
        mhat_fun = vector2Function(mhat, self.Vh[PARAMETER])
        mhatstar_fun = vector2Function(mhatstar, self.Vh[PARAMETER])
        phat_fun = vector2Function(phat, self.Vh[ADJOINT])
        uhatstar_fun = vector2Function(uhatstar, self.Vh[STATE])
        phatstar_fun = vector2Function(phatstar, self.Vh[STATE])

        f_form = self.residual(u, m, p, z)
        z_test = dl.TestFunction(self.Vh[OPTIMIZATION])

        # fwd
        dyr = dl.derivative(f_form, p, pstar_fun)
        dyzr = dl.derivative(dyr, z, z_test)
        dyzr = dl.assemble(dyzr)

        # adj
        dxr = dl.derivative(f_form, u, ustar_fun)
        dxzr = dl.derivative(dxr, z, z_test)
        dxzr = dl.assemble(dxzr)

        # incfwd
        dyr = dl.derivative(f_form, p, phatstar_fun)

        dyxr = dl.derivative(dyr, u, uhat_fun)
        dyxzr = dl.derivative(dyxr, z, z_test)
        dyxzr = dl.assemble(dyxzr)

        dymr = dl.derivative(dyr, m, mhat_fun)
        dymzr = dl.derivative(dymr, z, z_test)
        dymzr = dl.assemble(dymzr)

        # incadj
        dxr = dl.derivative(f_form, u, uhatstar_fun)

        dxyr = dl.derivative(dxr, p, phat_fun)
        dxyzr = dl.derivative(dxyr, z, z_test)
        dxyzr = dl.assemble(dxyzr)

        dxxr = dl.derivative(dxr, u, uhat_fun)
        dxxzr = dl.derivative(dxxr, z, z_test)
        dxxzr = dl.assemble(dxxzr)

        dxmr = dl.derivative(dxr, m, mhat_fun)
        dxmzr = dl.derivative(dxmr, z, z_test)
        dxmzr = dl.assemble(dxmzr)

        # hessian
        dmr = dl.derivative(f_form, m, mhatstar_fun)

        dmmr = dl.derivative(dmr, m, mhat_fun)
        dmmzr = dl.derivative(dmmr, z, z_test)
        dmmzr = dl.assemble(dmmzr)

        dmyr = dl.derivative(dmr, p, phat_fun)
        dmyzr = dl.derivative(dmyr, z, z_test)
        dmyzr = dl.assemble(dmyzr)

        dmxr = dl.derivative(dmr, u, uhat_fun)
        dmxzr = dl.derivative(dmxr, z, z_test)
        dmxzr = dl.assemble(dmxzr)

        dmxzq = self.generate_optimization()
        qoi.apply_ijk(PARAMETER,STATE,OPTIMIZATION,mhatstar,uhat,dmxzq)

        dmmzq = self.generate_optimization()
        qoi.apply_ijk(PARAMETER,PARAMETER,OPTIMIZATION,mhatstar,mhat,dmmzq)

        dxxzq = self.generate_optimization()
        qoi.apply_ijk(STATE,STATE,OPTIMIZATION,uhatstar,uhat,dxxzq)

        dxmzq = self.generate_optimization()
        qoi.apply_ijk(STATE,PARAMETER,OPTIMIZATION,uhatstar,mhat,dxmzq)

        return dyzr, dxzr, dyxzr, dymzr, dxyzr, dxxzr, dxmzr, dmmzr, dmyzr, dmxzr, dmxzq, dmmzq, dxxzq, dxmzq
