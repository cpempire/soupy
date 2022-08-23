from dolfin import *

import numpy as np


###########################################
class CahnHilliardProblem:

    def __init__(self, Vh, m, theta, rescaling=True, save=True, bcs=None):
        self.Vh = Vh
        self.m = m
        self.theta = theta
        self.rescaling = rescaling
        self.save = save
        self.kappa, self.epsilon, self.sigma = theta[0], theta[1], theta[2]
        self.bcs = bcs

        self.z = Function(Vh)
        self.z_trial = TrialFunction(Vh)
        self.z_test = TestFunction(Vh)
        self.delta_z = TrialFunction(Vh)
        self.z_save = Function(Vh)

        # build Msolver to compute gradient norm
        u, mu = split(self.z_trial)
        v, nu = split(self.z_test)
        M = assemble((u*v + mu*nu)*dx)
        self.Msolver = self._creat_solver(Vh)
        self.Msolver.set_operator(M)

        # choose solver for Newton linear system
        self.solver = self._creat_solver(Vh)

        self.help = self.generate_vector()
        self.mpi_rank = Vh.mesh().mpi_comm().rank
        # save data in file
        if self.save:
            filename = "data/u-full" + "_rescaling_" + str(self.rescaling) + ".pvd"
            self.file_u = File(filename, "compressed")

        # self.file_mu = File("data/mu-full.pvd", "compressed")

        # build poisson solver with homogeneous Neumann BC
        P1 = FiniteElement("Lagrange", Vh.mesh().ufl_cell(), 1)
        R = FiniteElement("Real", Vh.mesh().ufl_cell(), 0)
        Wh = FunctionSpace(Vh.mesh(), P1 * R)
        (w_trial, c) = TrialFunction(Wh)
        (self.w_test, d) = TestFunction(Wh)
        self.w = Function(Wh)
        a = (dot(grad(w_trial), grad(self.w_test)) + c * self.w_test + w_trial * d) * dx
        A = assemble(a)
        self.solver4poisson = self._creat_solver(Wh)
        self.solver4poisson.set_operator(A)

        # build solver to solve mu from given u
        self.Vh_u = FunctionSpace(Vh.mesh(), P1)
        u = TrialFunction(self.Vh_u)
        v = TestFunction(self.Vh_u)
        M = assemble(u*v*dx)
        self.solver4mu = self._creat_solver(self.Vh_u)
        self.solver4mu.set_operator(M)

    def _creat_solver(self, Vh):

        return PETScLUSolver(Vh.mesh().mpi_comm())

    def generate_vector(self):
        return Function(self.Vh).vector()

    def weak_form(self, z, z_test):
        u, mu = split(z)
        v, nu = split(z_test)

        # sigmainv = 1/self.sigma
        if self.rescaling:
            L0 = dot(grad(mu), grad(nu)) * dx + self.sigma * (u - self.m) * nu * dx
            L1 = self.epsilon * self.epsilon * dot(grad(u), grad(v)) * dx - mu * v * dx - self.kappa * (1 - u * u) * u * v * dx
            L = L0 + self.sigma * L1

            # L0 = dot(grad(mu), grad(v)) * dx + self.sigma * (u - self.m) * v * dx
            # L1 = mu * nu * dx + self.kappa * (1 - u * u) * u * nu * dx - self.epsilon * self.epsilon * dot(grad(u), grad(nu)) * dx
            # L = L0 - self.sigma * L1
        else:
            L0 = dot(grad(mu), grad(v)) * dx + self.sigma * (u - self.m) * v * dx
            L1 = mu * nu * dx + self.kappa * (1 - u * u) * u * nu * dx - self.epsilon * self.epsilon * dot(grad(u), grad(nu)) * dx
            L = L0 + L1

        return L

    def A(self, z):
        self.z.vector().set_local(z.get_local())
        form = self.weak_form(self.z, self.z_test)
        dzform = derivative(form, self.z, self.delta_z)
        A = assemble(dzform)
        if self.bcs is not None:
            [bc.apply(A) for bc in self.bcs]

        return A

    def A_modified(self, z):
        self.z.vector().set_local(z.get_local())
        form = self.weak_form(self.z, self.z_test)
        dzform = derivative(form, self.z, self.delta_z)

        delta_u, delta_mu = split(self.delta_z)
        v, nu = split(self.z_test)
        u, mu = split(self.z)

        modified_form = (1 - self.epsilon * np.sqrt(self.sigma)/self.kappa) * self.sigma * self.kappa * (1-u*u) * delta_u * v * dx

        A = assemble(dzform + modified_form)

        if self.bcs is not None:
            [bc.apply(A) for bc in self.bcs]

        return A

    def P(self, z):
        # implement preconditioner
        pass

    def gradient(self, z):
        self.z.vector().set_local(z.get_local())
        form = self.weak_form(self.z, self.z_test)
        g = assemble(form)
        if self.bcs is not None:
            [bc.apply(g) for bc in self.bcs]

        self.Msolver.solve(self.help, g)

        return g, np.sqrt(g.inner(self.help))

    def poisson_solve(self, z):
        # solve w given u
        self.z.vector().set_local(z.get_local())
        u, mu = split(self.z)
        u = project(u, self.Vh_u)
        L = (u - self.m) * self.w_test * dx
        # solve(a == L, self.w)
        b = assemble(L)
        self.solver4poisson.solve(self.w.vector(), b)
        (w, c) = self.w.split()

        return w

    def mu_solve(self, z):
        # solve mu given u for mu = self.kappa
        self.z.vector().set_local(z.get_local())
        u, mu = split(self.z)
        u = project(u, self.Vh_u)
        v = TestFunction(self.Vh_u)
        mu_rhs = assemble(self.epsilon * self.epsilon * dot(grad(u), grad(v)) * dx - self.kappa * (1 - u * u) * u * v * dx)
        mu = Function(self.Vh_u)
        self.solver4mu.solve(mu.vector(), mu_rhs)

        return mu

    def cost(self, z):
        # merit function for line search
        self.z.vector().set_local(z.get_local())
        u, mu = split(self.z)
        w = self.poisson_solve(z)
        energy_form = self.kappa*(1-u*u)*(1-u*u)*Constant(0.25) + \
                      0.5 * self.epsilon*self.epsilon*dot(grad(u), grad(u)) + \
                      0.5 * self.sigma * dot(grad(w), grad(w))  # (u - self.m) * w
        energy = assemble(energy_form * dx)

        # f1 = assemble(self.kappa*(1-u*u)*(1-u*u)*Constant(0.25) * dx)
        # f2 = assemble(self.epsilon*self.epsilon*dot(grad(u), grad(u)) * dx)
        # f3 = assemble(self.sigma * (u - self.m) * w * dx)
        # print("f1, f2, f3 = ", f1, f2, f3)

        return energy