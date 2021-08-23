from __future__ import absolute_import, division, print_function

import dolfin as dl
dl.dx = dl.dx(metadata={'quadrature_degree':2, "representation":'uflacs'})
import numpy as np

path = "../../"

import sys
sys.path.append(path)
from soupy.utils.variables import STATE, PARAMETER, ADJOINT, OPTIMIZATION
from soupy.utils.vector2function import vector2Function


class QoI(object):
    """
    define the quantity of interest and its derivative information
    """
    def __init__(self, mesh, Vh):
        """
        Constructor.
        INPUTS:
        - mesh: the mesh
        - Vh: the finite element space for [state, parameter, adjoint, optimization] variable
        """
        self.mesh = mesh
        self.Vh = Vh
        self.x = [dl.Function(Vh[STATE]).vector(), dl.Function(Vh[PARAMETER]).vector(),
                  dl.Function(Vh[ADJOINT]).vector(), dl.Function(Vh[OPTIMIZATION]).vector()]
        self.x_test = [dl.TestFunction(Vh[STATE]), dl.TestFunction(Vh[PARAMETER]),
                       dl.TestFunction(Vh[ADJOINT]), dl.TestFunction(Vh[OPTIMIZATION])]

    def form(self, u, m, z):
        """
        weak form of the qoi
        :param u: state
        :param m: parameter
        :param z: optimization
        :return: weak form
        """

        return None

    def eval(self, x):
        """
        evaluate the qoi at given x
        :param x: [state, parameter, adjoint, optimization] variable
        :return: qoi(x)
        """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])

        return dl.assemble(self.form(u, m, z))

    def adj_rhs(self, x, rhs):
        """
        The right hand for the adjoint problem (i.e. the derivative of the Lagrangian funtional
        with respect to the state u).

        INPUTS:
        - x coefficient vector of all variables
        - rhs: FEniCS vector to store the rhs for the adjoint problem.
        """
        self.grad_state(x, rhs)
        rhs *= -1

    def grad_state(self, x, g):
        """
        The partial derivative of the qoi with respect to the state variable.

        INPUTS:
        - x coefficient vector of all variables
        - g: FEniCS vector to store the gradient w.r.t. the state.
        """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])

        f_form = self.form(u, m, z)
        f_u = dl.assemble(dl.derivative(f_form, u, self.x_test[STATE]))

        g.zero()
        g.axpy(1.0, f_u)

    def grad_parameter(self, x, g):
        """
        The partial derivative of the qoi with respect to the parameter variable.

        INPUTS:
        - x coefficient vector of [state,parameter,adjoint,optimization] variable
        - g: FEniCS vector to store the gradient w.r.t. the parameter.
        """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])

        f_form = self.form(u, m, z)
        f_m = dl.assemble(dl.derivative(f_form, m, self.x_test[PARAMETER]))

        g.zero()
        g.axpy(1.0, f_m)

        # print("gm_obs", np.zeros(100), "gm_eva", g.get_local()[:100])
        #
        # g.zero()

    def grad_optimization(self, x, g):
        """
        The partial derivative of the qoi with respect to the optimization variable.

        INPUTS:
        - x coefficient vector of [state,parameter,adjoint,optimization] variable
        - g: FEniCS vector to store the gradient w.r.t. the optimization.
        """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        z = vector2Function(x[OPTIMIZATION], self.Vh[OPTIMIZATION])

        f_form = self.form(u, m, z)
        f_z = dl.assemble(dl.derivative(f_form, z, self.x_test[OPTIMIZATION]))

        g.zero()
        g.axpy(1.0, f_z)

        # print("gz_obs", np.zeros(100), "gz_eva", g.get_local()[:100])
        #
        # g.zero()

    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation \delta_ij (i,j = STATE,PARAMETER,OPTIMIZATION) of the q.o.i. in direction dir.

        INPUTS:
        - i,j integer (STATE=0, PARAMETER=1, OPTIMIZATION=3) which indicates with respect to which variables differentiate
        - dir the direction in which to apply the second variation
        - out: FEniCS vector to store the second variation in the direction dir.

        NOTE: setLinearizationPoint must be called before calling this method.
        """

        out.zero()

        x_fun = [vector2Function(self.x[s], self.Vh[s]) for s in range(len(self.x))]
        f_form = self.form(x_fun[STATE], x_fun[PARAMETER], x_fun[OPTIMIZATION])
        dir_fun = vector2Function(dir, self.Vh[j])

        f_i = dl.derivative(f_form, x_fun[i], self.x_test[i])
        f_ij = dl.derivative(f_i, x_fun[j], dir_fun)
        out.axpy(1.0, dl.assemble(f_ij))


    def apply_ijk(self,i,j,k,dir1,dir2, out):
        """
        Apply the third order variation of the q.o.i. w.r.t. ijk in direction dir1, dir2 for j and k
        :param i: STATE or PARAMETER or OPTIMIZATION
        :param j:
        :param k:
        :param dir1:
        :param dir2:
        :param out:
        :return: out
        """
        out.zero()

        x_fun = [vector2Function(self.x[s], self.Vh[s]) for s in range(len(self.x))]
        f_form = self.form(x_fun[STATE], x_fun[PARAMETER], x_fun[OPTIMIZATION])
        dir1_fun, dir2_fun = vector2Function(dir1, self.Vh[i]), vector2Function(dir2, self.Vh[j])

        f_i = dl.derivative(f_form, x_fun[i], dir1_fun)
        f_ij = dl.derivative(f_i, x_fun[j], dir2_fun)
        f_ijk = dl.derivative(f_ij, x_fun[k], self.x_test[k])
        out.axpy(1.0, dl.assemble(f_ijk))

        # print("fijk_obs", np.zeros(100), "fijk_eva", g.get_local()[:100])

    def setLinearizationPoint(self, x):
        """
        Specify the linearization point for computation of the second variations in method apply_ij.

        INPUTS:
        - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p
        """
        for i in range(len(x)):
            self.x[i].zero()
            self.x[i].axpy(1.0, x[i])


# class QoIObjective(QoI):
#
#     """
#     integral of the inflow rate along the boundary
#     q = \int_{\partial D} exp(m) dot (grad u, n) * ds
#     u is the state variable
#     """
#     def __init__(self, mesh, Vh):
#         """
#         Constructor.
#         INPUTS:
#         - mesh: the mesh
#         - Vh: the finite element space for [state, parameter, adjoint] variable
#         """
#         super(QoIObjective, self).__init__(mesh, Vh)
#
#         self.n = dl.FacetNormal(mesh)
#         self.Obs = dl.assemble(self.x_test[ADJOINT] * dl.dx)
#
#         mollifierlist = []
#         x = np.linspace(0.25, 0.75, 5)
#         y = np.linspace(0.25, 0.75, 5)
#         for i in range(5):
#             for j in range(5):
#                 mollifierlist.append(
#                     dl.Expression("exp(-(pow(x[0]-xi,2)+pow(x[1]-yj,2))/(pow(0.1,2)))/2", xi=x[i], yj=y[j], degree=2))
#
#         self.mollifier = dl.as_vector(mollifierlist)
#
#         mollifierMat = [ [ None for i in range(25) ] for j in range(25) ]
#         for i in range(25):
#             for j in range(25):
#                 if i == j:
#                     mollifierMat[i][j] = mollifierlist[i]
#                 else:
#                     mollifierMat[i][j] = 0.
#         self.mollifierMat = dl.as_matrix(mollifierMat)
#
#         z = 18.*np.ones(25)
#         z_target = dl.Function(self.Vh[OPTIMIZATION])
#         idx = z_target.vector().local_range()
#         z_target.vector().set_local(z[idx[0]:idx[1]])
#         self.z_target = z_target
#
#     def form(self, u, m, z):
#         """
#         weak form of the q.o.i.
#         :param x:
#         :return:
#         """
#
#         # f = - dl.exp(m) * dl.dot(dl.grad(u), self.n) * dl.ds
#
#         f = - dl.dot(dl.grad(u), self.n) * dl.ds
#
#         # f = u * dl.dx
#
#         # f = - dl.inner(self.mollifier, z)*dl.dx
#
#         # f = dl.dot(self.mollifierMat*(z-self.z_target), (z-self.z_target)) * dl.dx
#
#         # f = dl.inner(self.mollifier, (z-self.z_target)) * dl.inner(self.mollifier, (z-self.z_target))*dl.dx
#
#         return f


class QoIObjective(QoI):

    """
    integral of the inflow rate along the boundary
    q = \int_{\partial D} exp(m) dot (grad u, n) * ds
    u is the state variable
    """
    def __init__(self, mesh, Vh):
        """
        Constructor.
        INPUTS:
        - mesh: the mesh
        - Vh: the finite element space for [state, parameter, adjoint] variable
        """
        super(QoIObjective, self).__init__(mesh, Vh)

        self.num = 25

        z = 18.*np.ones(self.num)
        z_target = dl.Function(self.Vh[OPTIMIZATION]).vector()
        idx = z_target.local_range()
        z_target.set_local(z[idx[0]:idx[1]])
        self.z_target = z_target

        self.z_diff = dl.Function(Vh[OPTIMIZATION]).vector()

    def form(self, u, m, z):

        # arbitrary
        return u * dl.dx

    def eval(self, x):

        z = x[OPTIMIZATION]
        self.z_diff.zero()
        self.z_diff.axpy(1.0, z)
        self.z_diff.axpy(-1.0, self.z_target)

        return 1./self.num * self.z_diff.inner(self.z_diff)

    def adj_rhs(self, x, rhs):
        """
        The right hand for the adjoint problem (i.e. the derivative of the Lagrangian funtional
        with respect to the state u).

        INPUTS:
        - x coefficient vector of all variables
        - rhs: FEniCS vector to store the rhs for the adjoint problem.
        """
        self.grad_state(x, rhs)
        rhs *= -1

    def grad_state(self, x, g):
        """
        The partial derivative of the qoi with respect to the state variable.

        INPUTS:
        - x coefficient vector of all variables
        - g: FEniCS vector to store the gradient w.r.t. the state.
        """
        g.zero()

    def grad_parameter(self, x, g):
        """
        The partial derivative of the qoi with respect to the parameter variable.

        INPUTS:
        - x coefficient vector of [state,parameter,adjoint,optimization] variable
        - g: FEniCS vector to store the gradient w.r.t. the parameter.
        """
        g.zero()

    def grad_optimization(self, x, g):
        """
        The partial derivative of the qoi with respect to the optimization variable.

        INPUTS:
        - x coefficient vector of [state,parameter,adjoint,optimization] variable
        - g: FEniCS vector to store the gradient w.r.t. the optimization.
        """
        z = x[OPTIMIZATION]

        self.z_diff.zero()
        self.z_diff.axpy(1.0, z)
        self.z_diff.axpy(-1.0, self.z_target)

        g.zero()
        g.axpy(2./self.num, self.z_diff)

    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation \delta_ij (i,j = STATE,PARAMETER,OPTIMIZATION) of the q.o.i. in direction dir.

        INPUTS:
        - i,j integer (STATE=0, PARAMETER=1, OPTIMIZATION=3) which indicates with respect to which variables differentiate
        - dir the direction in which to apply the second variation
        - out: FEniCS vector to store the second variation in the direction dir.

        NOTE: setLinearizationPoint must be called before calling this method.
        """

        out.zero()

    def apply_ijk(self,i,j,k,dir1,dir2, out):
        """
        Apply the third order variation of the q.o.i. w.r.t. ijk in direction dir1, dir2 for j and k
        :param i: STATE or PARAMETER or OPTIMIZATION
        :param j:
        :param k:
        :param dir1:
        :param dir2:
        :param out:
        :return: out
        """
        out.zero()

    def setLinearizationPoint(self, x):
        """
        Specify the linearization point for computation of the second variations in method apply_ij.

        INPUTS:
        - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p
        """
        for i in range(len(x)):
            self.x[i].zero()
            self.x[i].axpy(1.0, x[i])




# ############################################################################
# class QoIConstraint:
#
#     """
#     misfit term of simulaation-observation at a few locations
#     Q = ||O(u) - \bar{u}||^2
#     O is the observation functional
#     u is the state variable
#     \bar{u} is the target
#     """
#     def __init__(self, mesh, Vh):
#         """
#         Constructor.
#         INPUTS:
#         - mesh: the mesh
#         - Vh_STATE: the finite element space for the state variable
#         """
#         num = 5
#         self.num = num
#         x = np.linspace(0.25, 0.75, num)
#         y = np.linspace(0.25, 0.75, num)
#         xv,yv = np.meshgrid(x,y)
#         ubar = -0 + 0. * (1 - 2*(xv-0.5)**2 - 2*(yv-0.5)**2)
#         self.ubar = np.reshape(ubar, num**2, 1)
#
#         self.Obs = dict()
#         for i in range(num):
#             for j in range(num):
#                 obs = dl.Function(Vh[STATE]).vector()
#                 ps = dl.PointSource(Vh[STATE], dl.Point(x[i], y[j]), 1.)
#                 ps.apply(obs)
#                 self.Obs[i*num+j] = obs
#
#         self.x = [dl.Function(Vh[i]).vector() for i in range(4)]
#         self.Vh = Vh
#
#     def form(self, x):
#         """
#         Build the weak form of the qoi
#         :param x:
#         :return:
#         """
#         x_fun = [None] * len(x)
#         for i in range(len(x)):
#             x_fun[i] = vector2Function(x[i], self.Vh[i])
#
#         # this is not a correct representation, just a place holder
#         v = dl.TestFunction(self.Vh[STATE])
#         return x_fun[STATE] * dl.dx
#
#     def eval(self, x):
#         """
#         Evaluate the quantity of interest at a given point in the state and
#         parameter space.
#
#         INPUTS:
#         - x coefficient vector of [state,parameter,adjoint,optimization] variable
#         """
#         QoI = -2.
#         for i in range(self.num**2):
#             QoI += (self.Obs[i].inner(x[STATE]) - self.ubar[i])**2/25
#
#         # print("QoI", QoI)
#
#         return QoI
#
#     def adj_rhs(self,x,rhs):
#         """
#         The right hand for the adjoint problem (i.e. the derivative of the Lagrangian funtional
#         with respect to the state u).
#
#         INPUTS:
#         - x coefficient vector of [state,parameter,adjoint,optimization] variable
#         - rhs: FEniCS vector to store the rhs for the adjoint problem.
#         """
#         ### rhs = - df/dstate
#         self.grad_state(x, rhs)
#         rhs *= -1
#
#     def grad_state(self,x,g):
#         """
#         The partial derivative of the qoi with respect to the state variable.
#
#         INPUTS:
#         - x coefficient vector of [state,parameter,adjoint,optimization] variable
#         - g: FEniCS vector to store the gradient w.r.t. the state.
#         """
#         g.zero()
#         for i in range(self.num**2):
#             g.axpy(2./25*(self.Obs[i].inner(x[STATE]) - self.ubar[i]), self.Obs[i])
#
#     def grad_parameter(self, x, g):
#         """
#         The partial derivative of the qoi with respect to the parameter variable.
#
#         INPUTS:
#         - x coefficient vector of [state,parameter,adjoint,optimization] variable
#         - g: FEniCS vector to store the gradient w.r.t. the parameter.
#         """
#         g.zero()
#
#     def grad_optimization(self, x, g):
#         """
#         The partial derivative of the qoi with respect to the optimization variable.
#
#         INPUTS:
#         - x coefficient vector of [state,parameter,adjoint,optimization] variable
#         - g: FEniCS vector to store the gradient w.r.t. the optimization.
#         """
#         g.zero()
#
#     def apply_ij(self,i,j, dir, out):
#         """
#         Apply the second variation \delta_ij (i,j = STATE,PARAMETER,OPTIMIZATION) of the q.o.i. in direction dir.
#
#         INPUTS:
#         - i,j integer (STATE=0, PARAMETER=1, OPTIMIZATION=3) which indicates with respect to which variables differentiate
#         - dir the direction in which to apply the second variation
#         - out: FEniCS vector to store the second variation in the direction dir.
#
#         NOTE: setLinearizationPoint must be called before calling this method.
#         """
#         out.zero()
#         if i == STATE and j == STATE:
#             for n in range(self.num**2):
#                 out.axpy(2./25*self.Obs[n].inner(dir), self.Obs[n])
#
#     def apply_ijk(self,i,j,k,dir1,dir2, out):
#         """
#         Apply the third variation \delta_ijk (i,j,k = STATE,PARAMETER,OPTIMIZATION) of the q.o.i. in direction dir1, dir2
#
#         INPUTS:
#         - i,j,k integer (STATE=0, PARAMETER=1, OPTIMIZATION=3) which indicates with respect to which variables differentiate
#         - dir1,dir2 the direction in which to apply the second and third variations
#         - out: FEniCS vector to store the second variation in the direction dir.
#
#         NOTE: setLinearizationPoint must be called before calling this method.
#         """
#         out.zero()
#
#     def setLinearizationPoint(self, x):
#         """
#         Specify the linearization point for computation of the variations in method apply_ij and apply_ijk.
#
#         INPUTS:
#         - x coefficient vector of [state,parameter,adjoint,optimization] variable
#         """
#         for i in range(len(x)):
#             self.x[i].zero()
#             self.x[i].axpy(1., x[i])


class QoIConstraint(QoI):

    def __init__(self, mesh, Vh):
        super(QoIConstraint, self).__init__(mesh, Vh)

        # self.f_c = dl.interpolate(dl.Constant(1.5**2), Vh[STATE])
        self.f_c = dl.interpolate(dl.Constant(2), Vh[STATE])
        self.chi = dl.Expression("4.*(x[0]<=0.75)*(x[0]>=0.25)*(x[1]<=0.75)*(x[1]>=0.25)", degree=1)
        # self.u_tar = dl.Constant(0.)  # dl.Expression("-1.4 * (1. - 2*pow(x[0] - 0.5, 2) - 2*pow(x[1]-0.5, 2))", degree=2)

    def form(self, u, m, z):

        f = self.chi * u * u * dl.dx - self.f_c * dl.dx

        # f = self.chi * (u - self.u_tar) * (u - self.u_tar) * dl.dx - self.f_c * dl.dx

        return f


# class QoIConstraint(QoI):
#
#     def __init__(self, mesh, Vh):
#         super(QoIConstraint, self).__init__(mesh, Vh)
#
#         self.f_c = dl.interpolate(dl.Constant(-1.5), Vh[STATE])
#         self.chi = dl.Expression("4.*(x[0]<=0.75)*(x[0]>=0.25)*(x[1]<=0.75)*(x[1]>=0.25)", degree=1)
#
#     def form(self, u, m, z):
#
#         f = self.f_c * dl.dx - self.chi * u * dl.dx
#
#         return f

# class QoIConstraint:
#
#     """
#     Constraint function
#     Q = O(u)
#     O is the observation functional
#     u is the state variable
#     """
#     def __init__(self, mesh, Vh):
#         """
#         Constructor.
#         INPUTS:
#         - mesh: the mesh
#         - Vh: the finite element space for [state, parameter, adjoint] variable
#         """
#         self.Vh = Vh
#         self.v = dl.TestFunction(self.Vh[STATE])
#         self.chi = dl.Expression("4.*(x[0]<=0.75)*(x[0]>=0.25)*(x[1]<=0.75)*(x[1]>=0.25)", degree=1)
#
#         self.f_c = -1.5
#
#         self.Obs = dl.assemble(self.chi*self.v*dl.dx)
#         self.x = [dl.Function(Vh[STATE]).vector(), dl.Function(Vh[PARAMETER]).vector(),
#                   dl.Function(Vh[ADJOINT]).vector(), dl.Function(Vh[OPTIMIZATION]).vector()]
#
#     def form(self, u, m, z):
#         """
#         Build the weak form of the qoi
#         :param x:
#         :return:
#         """
#         # defined upto a constant
#         f = -self.chi*u*dl.dx
#
#         return f
#
#     def eval(self, x):
#         """
#         Evaluate the quantity of interest at a given point in the state and
#         parameter space.
#
#         INPUTS:
#         - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p
#         """
#
#         return self.f_c - self.Obs.inner(x[STATE])
#
#     def adj_rhs(self,x,rhs):
#         """
#         The right hand for the adjoint problem (i.e. the derivative of the Lagrangian funtional
#         with respect to the state u).
#
#         INPUTS:
#         - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p
#         - rhs: FEniCS vector to store the rhs for the adjoint problem.
#         """
#         ### rhs = - df/dstate
#         self.grad_state(x, rhs)
#         rhs *= -1
#
#     def grad_state(self,x,g):
#         """
#         The partial derivative of the qoi with respect to the state variable.
#
#         INPUTS:
#         - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p
#         - g: FEniCS vector to store the gradient w.r.t. the state.
#         """
#         g.zero()
#         g.axpy(-1.0, self.Obs)
#
#     def grad_parameter(self, x, g):
#         """
#         The partial derivative of the qoi with respect to the parameter variable.
#
#         INPUTS:
#         - x coefficient vector of [state,parameter,adjoint,optimization] variable
#         - g: FEniCS vector to store the gradient w.r.t. the parameter.
#         """
#         g.zero()
#
#     def grad_optimization(self, x, g):
#         """
#         The partial derivative of the qoi with respect to the optimization variable.
#
#         INPUTS:
#         - x coefficient vector of [state,parameter,adjoint,optimization] variable
#         - g: FEniCS vector to store the gradient w.r.t. the optimization.
#         """
#         g.zero()
#
#     def apply_ij(self,i,j, dir, out):
#         """
#         Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the q.o.i. in direction dir.
#
#         INPUTS:
#         - i,j integer (STATE=0, PARAMETER=1) which indicates with respect to which variables differentiate
#         - dir the direction in which to apply the second variation
#         - out: FEniCS vector to store the second variation in the direction dir.
#
#         NOTE: setLinearizationPoint must be called before calling this method.
#         """
#
#         out.zero()
#
#     def apply_ijk(self,i,j,k,dir1,dir2, out):
#         """
#         Apply the third order variation of the q.o.i. w.r.t. ijk in direction dir1, dir2 for j and k
#         :param i:
#         :param j:
#         :param k:
#         :param dir1:
#         :param dir2:
#         :param out:
#         :return:
#         """
#         out.zero()
#
#     def setLinearizationPoint(self, x):
#         """
#         Specify the linearization point for computation of the second variations in method apply_ij.
#
#         INPUTS:
#         - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p
#         """
#         for i in range(len(x)):
#             self.x[i].zero()
#             self.x[i].axpy(1.0, x[i])
