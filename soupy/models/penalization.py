from __future__ import absolute_import, division, print_function

import dolfin as dl
dl.dx = dl.dx(metadata={'quadrature_degree': 3, "representation":'uflacs'})

# import sys
# sys.path.append('../')
from ..utils.vector2function import vector2Function


class Penalization(object):

    def eval(self, z):
        raise NotImplementedError("Child class should implement method init_vector")

    def grad(self, z, out):
        raise NotImplementedError("Child class should implement method init_vector")

    def hessian(self, z, dir, out):
        raise NotImplementedError("Child class should implement method init_vector")


class L2Penalization(Penalization):

    def __init__(self, Vh,  dX, alpha, region=None):
        """
        :param Vh: function space for control variable
        :param dX: integral measure, e.g. dl.dx for domain, dl.ds for boundary
        :param alpha: scaling parameter
        :param region: penalization region
        :return:
        """
        self.Vh = Vh
        self.alpha = alpha
        self.dX = dX
        self.region = region

        z_trial = dl.TrialFunction(Vh)
        z_test = dl.TestFunction(Vh)
        self.z_fun = dl.Function(Vh)

        if region is None:
            self.Z = dl.assemble(alpha*dl.dot(z_trial,z_test)*dX)
        else:
            self.Z = dl.assemble(alpha*dl.dot(z_trial,z_test)*region*dX)

        self.Zz = dl.Function(Vh).vector()

        dlversion = (dl.DOLFIN_VERSION_MAJOR, dl.DOLFIN_VERSION_MINOR, dl.DOLFIN_VERSION_MICRO)
        if dlversion <= (1, 6, 0):
            # self.Hsolver = dl.PETScKrylovSolver("cg")
            self.Hsolver = dl.PETScLUSolver()
        else:
            # self.Hsolver = dl.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg")
            self.Hsolver = dl.PETScLUSolver(self.Vh.mesh().mpi_comm())

        self.Hsolver.set_operator(self.Z)

    def form(self, z_fun):
        if self.region is None:
            dl.Constant(self.alpha)*dl.dot(z_fun, z_fun)*self.dX
        else:
            dl.Constant(self.alpha)*dl.dot(z_fun, z_fun)*self.region*self.dX

    def eval(self, z):

        self.Z.mult(z, self.Zz)

        return z.inner(self.Zz)

    def grad(self, z, out):

        self.Z.mult(z, self.Zz)
        out.axpy(2., self.Zz)

    def hessian(self, z, dir, out):
        self.Z.mult(dir, self.Zz)
        out.axpy(2., self.Zz)

    def solve(self, z, b, identity=False):
        if identity:
            z.zero()
            z.axpy(1.0, b)
        else:
            self.Hsolver.solve(z, b)


class H1Penalization(Penalization):

    def __init__(self, Vh,  dX, alpha, region=None):
        """
        :param Vh: function space for control variable
        :param dX: integral measure, e.g. dl.dx for domain, dl.ds for boundary
        :param alpha: scaling parameter
        :param region: penalization region
        :return:
        """
        self.Vh = Vh
        self.alpha = alpha
        self.dX = dX
        self.region = region

        z_trial = dl.TrialFunction(Vh)
        z_test = dl.TestFunction(Vh)

        if region is None:
            self.Z = dl.assemble(alpha*dl.inner(dl.grad(z_trial),dl.grad(z_test))*dX)
        else:
            self.Z = dl.assemble(alpha*dl.inner(dl.grad(z_trial),dl.grad(z_test))*region*dX)

        self.Zz = dl.Function(Vh).vector()

        dlversion = (dl.DOLFIN_VERSION_MAJOR, dl.DOLFIN_VERSION_MINOR, dl.DOLFIN_VERSION_MICRO)
        if dlversion <= (1, 6, 0):
            self.Hsolver = dl.PETScKrylovSolver("cg")
        else:
            self.Hsolver = dl.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg")

        self.Hsolver.set_operator(self.Z)

    def form(self, z_fun):
        if self.region is None:
            dl.Constant(self.alpha)*dl.inner(dl.grad(z_fun),dl.grad(z_fun))*self.dX
        else:
            dl.Constant(self.alpha)*dl.inner(dl.grad(z_fun),dl.grad(z_fun))*self.region*self.dX

    def eval(self, z):

        self.Z.mult(z, self.Zz)

        return z.inner(self.Zz)

    def grad(self, z, out):

        self.Z.mult(z, self.Zz)
        out.axpy(2., self.Zz)

    def hessian(self, z, dir, out):
        self.Z.mult(dir, self.Zz)
        out.axpy(2., self.Zz)

    def solve(self, z, b, identity=False):
        if identity:
            z.zero()
            z.axpy(1.0, b)
        else:
            self.Hsolver.solve(z, b)


class L1Penalization(Penalization):

    def __init__(self, Vh,  dX, alpha, region=None):
        """
        :param Vh: function space for control variable
        :param dX: integral measure, e.g. dl.dx for domain, dl.ds for boundary
        :param alpha: scaling parameter
        :param region: penalization region
        :return:
        """
        self.Vh = Vh
        self.alpha = alpha
        self.dX = dX
        self.region = region

        self.z_fun = dl.Function(self.Vh)
        self.z_trial = dl.TrialFunction(self.Vh)
        self.z_test = dl.TestFunction(self.Vh)

        dlversion = (dl.DOLFIN_VERSION_MAJOR, dl.DOLFIN_VERSION_MINOR, dl.DOLFIN_VERSION_MICRO)
        if dlversion <= (1, 6, 0):
            self.Hsolver = dl.PETScKrylovSolver("cg")
        else:
            self.Hsolver = dl.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg")

    def form(self, z_fun):

        if self.region is None:
            return dl.Constant(self.alpha)*pow(z_fun*z_fun+1e-8,0.5)*self.dX
        else:
            return dl.Constant(self.alpha)*pow(z_fun*z_fun+1e-8,0.5)*self.region*self.dX

    def eval(self, z):

        z_fun = vector2Function(z, self.Vh)

        return dl.assemble(self.form(z_fun))

    def grad(self, z, out):

        z_fun = vector2Function(z, self.Vh)
        form = self.form(z_fun)

        z_test = dl.TestFunction(self.Vh)
        dz = dl.derivative(form, z_fun, z_test)

        out.axpy(1., dl.assemble(dz))

    def hessian(self, z, dir, out):
        z_fun = vector2Function(z, self.Vh)
        dir_fun = vector2Function(dir, self.Vh)
        form = self.form(z_fun)

        z_test = dl.TestFunction(self.Vh)
        dz = dl.derivative(form, z_fun, z_test)
        dzz = dl.derivative(dz, z_fun, dir_fun)

        out.axpy(1., dl.assemble(dzz))

    def solve(self, z, b, identity=True):
        if identity:
            z.zero()
            z.axpy(1.0, b)
        else:
            Grad = dl.derivative(self.form(self.z_fun), self.z_fun, self.z_trial)
            Hess = dl.derivative(Grad, self.z_fun, self.z_test)
            H0 = dl.assemble(Hess)
            self.Hsolver.set_operator(H0)
            self.Hsolver.solve(z, b)

    def setPoint(self, z):
        self.z_fun.vector().set_local(z.get_local())  # = vector2Function(z, self.penalization.Vh)


class L1PenalizationMultiVariable(Penalization):

    def __init__(self, Vh,  dX, alpha, region=None, dim=1):
        """
        :param Vh: function space for control variable
        :param dX: integral measure, e.g. dl.dx for domain, dl.ds for boundary
        :param alpha: scaling parameter
        :param region: penalization region
        :return:
        """
        self.Vh = Vh
        self.alpha = alpha
        self.dX = dX
        self.dim = dim
        self.region = region

        self.z_fun = dl.Function(self.Vh)
        self.z_trial = dl.TrialFunction(self.Vh)
        self.z_test = dl.TestFunction(self.Vh)

        dlversion = (dl.DOLFIN_VERSION_MAJOR, dl.DOLFIN_VERSION_MINOR, dl.DOLFIN_VERSION_MICRO)
        if dlversion <= (1, 6, 0):
            self.Hsolver = dl.PETScKrylovSolver("cg")
        else:
            self.Hsolver = dl.PETScKrylovSolver(self.Vh.mesh().mpi_comm(), "cg")

    def form(self, z_fun):
        f = [None]*self.dim
        c = [None]*self.dim
        for i in range(self.dim):
            zi = z_fun.sub(i,deepcopy=True)
            f[i] = pow(zi*zi+1e-8, 0.5)
            c[i] = dl.Constant(self.alpha)
        f_vec = dl.as_vector(f)
        c_vec = dl.as_vector(c)

        if self.region is None:
            return dl.dot(c_vec, f_vec)*self.dX
        else:
            return dl.dot(c_vec, f_vec)*self.region*self.dX

    def eval(self, z):

        z_fun = vector2Function(z, self.Vh)

        return dl.assemble(self.form(z_fun))

    def grad(self, z, out):

        z_fun = vector2Function(z, self.Vh)
        form = self.form(z_fun)

        z_test = dl.TestFunction(self.Vh)
        dz = dl.derivative(form, z_fun, z_test)

        out.axpy(1., dl.assemble(dz))

    def hessian(self, z, dir, out):
        z_fun = vector2Function(z, self.Vh)
        dir_fun = vector2Function(dir, self.Vh)
        form = self.form(z_fun)

        z_test = dl.TestFunction(self.Vh)
        dz = dl.derivative(form, z_fun, z_test)
        dzz = dl.derivative(dz, z_fun, dir_fun)

        out.axpy(1., dl.assemble(dzz))

    def solve(self, z, b, identity=True):
        if identity:
            z.zero()
            z.axpy(1.0, b)
        else:
            Grad = dl.derivative(self.form(self.z_fun), self.z_fun, self.z_trial)
            Hess = dl.derivative(Grad, self.z_fun, self.z_test)
            H0 = dl.assemble(Hess)
            self.Hsolver.set_operator(H0)
            self.Hsolver.solve(z, b)

    def setPoint(self, z):
        self.z_fun.vector().set_local(z.get_local()) # = vector2Function(z, self.penalization.Vh)


class L2PenalizationFiniteDimension(Penalization):

    def __init__(self, Vh,  dX, alpha, region=None):
        self.Vh = Vh
        self.alpha = alpha

    def eval(self, z):

        return self.alpha * z.inner(z)

    def grad(self, z, out):

        out.axpy(2.*self.alpha, z)

    def hessian(self, z, dir, out):

        out.axpy(2.*self.alpha, dir)

    def solve(self, z, b, identity=True):
        if identity:
            z.zero()
            z.axpy(1.0, b)


class H0inv:
    """
    used for bfgs algorithm, as initial Hessian inverse
    """
    def __init__(self, penalization):
        self.penalization = penalization
        self.z_fun = dl.Function(self.penalization.Vh)
        self.z_trial = dl.TrialFunction(self.penalization.Vh)
        self.z_test = dl.TestFunction(self.penalization.Vh)

        # print("self.penalization.Vh.family", self.penalization.Vh.family)
        # if self.penalization.Vh.family is not "R":
        dlversion = (dl.DOLFIN_VERSION_MAJOR, dl.DOLFIN_VERSION_MINOR, dl.DOLFIN_VERSION_MICRO)
        if dlversion <= (1,6,0):
            self.Hsolver = dl.PETScKrylovSolver("cg")
        else:
            self.Hsolver = dl.PETScKrylovSolver(self.penalization.Vh.mesh().mpi_comm(), "cg")

    def solve(self, z, b):
        # # if self.penalization.Vh.family is not "R":
        # Grad = dl.derivative(self.penalization.form(self.z_fun), self.z_fun, self.z_trial)
        # Hess = dl.derivative(Grad, self.z_fun, self.z_test)
        # H0 = dl.assemble(Hess)
        # # H0.mult(z_help, z)
        #
        # # pass
        # self.Hsolver.set_operator(H0)
        # self.Hsolver.solve(z, b)

        self.penalization.solve(z, b)

    def setPoint(self, z):
        self.penalization.z_fun.vector().set_local(z.get_local()) # = vector2Function(z, self.penalization.Vh)