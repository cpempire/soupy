from __future__ import absolute_import, division, print_function

import dolfin as dl
dl.dx = dl.dx(metadata={'quadrature_degree': 3, "representation":'uflacs'})

STATE = 0

class QoIHelmholtz:

    """
    misfit term of simulaation-observation at a few locations
    Q = ||O(u) - \bar{u}||^2
    O is the observation functional
    u is the state variable
    \bar{u} is the target
    """
    def __init__(self, mesh, Vh, chi_obs, u_obs):
        """
        Constructor.
        INPUTS:
        - mesh: the mesh
        - Vh: the finite element space for the state variable
        """
        self.mesh = mesh
        self.u_obs = u_obs.vector()
        v = dl.TestFunction(Vh)
        vr, vi = dl.split(v)
        # chi = dl.Expression("( ( pow(x[0]-10.,2)+pow(x[1]-5.,2) ) > 16 )", degree=1)
        self.chi = chi_obs
        self.obs = dl.assemble(vr*chi_obs*dl.dx) + dl.assemble(vi*chi_obs*dl.dx)

        self.state = dl.Function(Vh).vector()
        self.Vh = Vh

    def form(self, x_fun):
        xr_fun, xi_fun = dl.split(x_fun) # x_fun.split(True)
        u_fun = dl.Function(self.Vh, self.u_obs)
        ur_fun, ui_fun = u_fun.split(True)

        qoi_form = self.chi*((xr_fun-ur_fun)**2+(xi_fun-ui_fun)**2)*dl.dx

        return qoi_form

    def eval(self, x):
        """
        Evaluate the quantity of interest at a given point in the state and
        parameter space.

        INPUTS:
        - x coefficient vector of state variable
        """

        x_fun = dl.Function(self.Vh, x)
        u_fun = dl.Function(self.Vh, self.u_obs)
        xr_fun, xi_fun = x_fun.split(True)
        ur_fun, ui_fun = u_fun.split(True)

        QoI = dl.assemble(self.chi*((xr_fun-ur_fun)**2+(xi_fun-ui_fun)**2)*dl.dx)

        # diff2 = dl.Function(self.Vh).vector()
        # diff2_value = (x.array() - self.u_obs.array())**2
        # diff2.set_local(diff2_value)
        # QoI = self.obs.inner(diff2)

        # QoI = self.obs.inner(x)

        return QoI

    def adj_rhs(self,x,rhs):
        """
        The right hand for the adjoint problem (i.e. the derivative of the Lagrangian funtional
        with respect to the state u).

        INPUTS:
        - x coefficient vector of state variable
        - rhs: FEniCS vector to store the rhs for the adjoint problem.
        """
        ### rhs = - df/dstate
        self.grad_state(x, rhs)
        rhs *= -1

    def grad_state(self,x,g):
        """
        The partial derivative of the qoi with respect to the state variable.

        INPUTS:
        - x coefficient vector of state variable
        - g: FEniCS vector to store the gradient w.r.t. the state.
        """

        x_fun = dl.Function(self.Vh, x)
        u_fun = dl.Function(self.Vh, self.u_obs)
        xr_fun, xi_fun = x_fun.split(True)
        ur_fun, ui_fun = u_fun.split(True)
        x_test = dl.TestFunction(self.Vh)
        xr_test, xi_test = dl.split(x_test)

        g.zero()
        g.axpy(2., dl.assemble(self.chi*((xr_fun-ur_fun)*xr_test + (xi_fun-ui_fun)*xi_test)*dl.dx))

        # diff = dl.Function(self.Vh).vector()
        # diff_value = 2*(x.array() - self.u_obs.array())
        # diff.set_local(diff_value)
        # g.zero()
        # g.axpy(self.obs.inner(diff), self.obs)

        # g.zero()
        # g.axpy(1., self.obs)

    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the q.o.i. in direction dir.

        INPUTS:
        - i,j integer (STATE=0, PARAMETER=1) which indicates with respect to which variables differentiate
        - dir the direction in which to apply the second variation
        - out: FEniCS vector to store the second variation in the direction dir.

        NOTE: setLinearizationPoint must be called before calling this method.
        """
        x_test = dl.TestFunction(self.Vh)
        xr_test, xi_test = dl.split(x_test)
        dir_fun = dl.Function(self.Vh, dir)
        dir_fun_r, dir_fun_i = dir_fun.split(True)
        out.zero()
        if i == STATE and j == STATE:
            out.axpy(2.,dl.assemble(self.chi*(dir_fun_r*xr_test + dir_fun_i*xi_test)*dl.dx))
            # out.axpy(2*self.obs.inner(dir), self.obs)
            # out.axpy(0., self.obs)

    def apply_ijk(self,i,j,k,dir1,dir2, out):
        ## Q_xxx(dir1, dir2, x_test)
        out.zero()
        if i == STATE and j == STATE and k == STATE:
            out.axpy(0., self.obs)

    def setLinearizationPoint(self, x):
        """
        Specify the linearization point for computation of the second variations in method apply_ij.

        INPUTS:
        - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p
        """
        self.state.zero()
        self.state.axpy(1., x)
