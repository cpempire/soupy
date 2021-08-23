from __future__ import absolute_import, division, print_function

import numpy as np
import dolfin as dl
dl.dx = dl.dx(metadata={'quadrature_degree':2, "representation":'uflacs'}) #, "representation":'uflacs'


class CostFunctional(object):

    def objective(self):
        """ return the evaluation of the objective function """
        raise NotImplementedError("Child class should implement method objective")

    def costValue(self, z):
        """ return the evaluation of the cost functional """
        raise NotImplementedError("Child class should implement method costValue")

    def costGradient(self, z):
        """ return the gradient of the cost functional """
        raise NotImplementedError("Child class should implement method costGradient")

    def costHessian(self, z, z_hat, FD=False):
        """ return the Hessian action in z_hat of the cost functional """
        raise NotImplementedError("Child class should implement method costHessian")

    def checkGradient(self, z, dlcomm=None, plotFlag=True):

        n_eps = 5
        eps = np.power(.1, np.arange(n_eps, 0, -1))
        errGrad = np.zeros(n_eps)

        cost = self.costValue(z)
        dz, dznorm = self.costGradient(z)
        # print ("cost", cost, "dz", np.linalg.norm(dz))
        if isinstance(z, np.ndarray):
            sigma = np.amax(np.absolute(z))
            if sigma <= 0:
                sigma = 1.
            if dlcomm is None:
                dir = np.random.normal(0.,sigma,z.shape[0])
            else:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                dir = dl.Vector(dlcomm, z.shape[0])
                dir.set_local(np.random.normal(0.,sigma,dir.local_size()))
                dir_array = dir.gather_on_zero()
                dir = comm.bcast(dir_array,root=0)
            dzdir = np.dot(dir,dz)
        else:
            dir = dl.Vector(z.copy())
            dir.zero()
            sigma = np.amax(np.absolute(z.get_local()))
            if sigma <= 0:
                sigma = 1.
            dir.set_local(np.random.normal(0., sigma, z.get_local().shape[0]))
            dzdir = dir.inner(dz)

        if dlcomm is None:
            mpi_rank = 0
        else:
            mpi_rank = dl.MPI.rank(dlcomm)

        for i in range(n_eps):
            if isinstance(z, np.ndarray):
                zplus = z + eps[i]*dir

            else:
                zplus = dl.Vector(z.copy())
                zplus.axpy(eps[i], dir)

            costplus = self.costValue(zplus)

            if mpi_rank == 0:
                # print ("(costplus-cost)/eps[i]", (costplus-cost)/eps[i], "np.dot(dir,dz)", np.dot(dir,dz))
                errGrad[i] = np.abs(((costplus-cost)/eps[i] - dzdir)/dzdir)
                print("check Gradient:  epsilon = ", eps[i], ", relative error = ", errGrad[i])

        if plotFlag and mpi_rank == 0:
            self.checkPlotErrors(eps, errGrad)
        return eps, errGrad

    def checkHessian(self, z, dlcomm=None, plotFlag=True):

        n_eps = 7
        eps = np.power(.1, np.arange(n_eps, 0, -1))
        errHess = np.zeros(n_eps)

        self.costValue(z)
        dz = self.costGradient(z)

        if isinstance(z, np.ndarray):
            if dlcomm is None:
                dir = np.random.normal(0., 1., z.shape[0])
            else:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                dir = dl.Vector(dlcomm, z.shape[0])
                dir.set_local(np.random.normal(0., 1., dir.local_size()))
                dir_array = dir.gather_on_zero()
                dir = comm.bcast(dir_array,root=0)
        else:
            dir = dl.Vector(z.copy())
            dir.zero()
            dir.set_local(np.random.normal(0., 1., z.get_local().shape[0]))

        dzz = self.costHessian(z, dir)

        if dlcomm is None:
            mpi_rank = 0
        else:
            mpi_rank = dl.MPI.rank(dlcomm)

        for i in range(n_eps):
            if isinstance(z, np.ndarray):
                zplus = z + eps[i]*dir
            else:
                zplus = dl.Vector(z.copy())
                zplus.axpy(eps[i], dir)

            self.costValue(zplus)
            dzplus = self.costGradient(zplus)
            if mpi_rank == 0:
                if isinstance(z, np.ndarray):
                    errHess[i] = np.linalg.norm((dzplus-dz)/eps[i] - dzz)
                else:
                    dzplus.axpy(-1.0, dz)
                    dzplus[:] = dzplus[:]/eps[i]
                    dzplus.axpy(-1.0, dzz)
                    errHess[i] = np.sqrt(dzplus.inner(dzplus))
                print("check Hessian:  epsilon = ", eps[i], ", ||error||_2 = ", errHess[i])

        if plotFlag and mpi_rank == 0:
            self.checkPlotErrors(eps, errHess)

        return eps, errHess

    def checkPlotErrors(self, eps, error):
        try:
            import matplotlib.pyplot as plt
        except:
            print( "Matplotlib is not installed.")
            return
        print("what happed here 1")
        plt.figure()
        print("what happed here 2")
        plt.loglog(eps, error, "-ob")
        print("what happed here 3")
        # try:
        #     plt.loglog(eps, error, "-ob", eps, eps*(error[0]/eps[0]), "-.k")
        # except:
        #     plt.semilogx(eps, error, "-ob")
        plt.title("Finite Difference Check")
        plt.xlabel("epsilon")
        plt.ylabel("error")
        print("what happed here 4")
        plt.show()