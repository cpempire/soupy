# compute the trace by randomized MC and randomized SVD

from construction import *

plotFlag = False

if plotFlag:
    import matplotlib.pyplot as plt

################### 6. test the trace estimation #########################
N_tr = 100

type_set = ['random', 'constant', 'linear', 'quadratic', 'saa']
for test in range(5):
    for n_pde in range(N_pde):
        x = pde[n_pde].generate_state()
        m = prior.mean
        y = pde[n_pde].generate_state()
        z = pde[n_pde].generate_optimization()

        #################################################################################
        tol = 1.e-10
        ## solve the forward pde at given phi
        if test == 0:
            if mpi_rank == 0:
                print("="*10+"at random control"+"="*10)
            z = pde[n_pde].generate_optimization()
            z.set_local(np.random.normal(0, 0.1, z.get_local().shape[0]))
        else:
            type = type_set[test]
            if mpi_rank == 0:
                print("="*10+"at optimal control with "+type+" approximation"+"="*10)

            z = dl.Function(Vh_OPTIMIZATION)
            input_file = dl.HDF5File(mesh.mpi_comm(), "data/"+type+"/z_opt_load.h5",'r')
            input_file.read(z, 'z_opt')
            input_file.close()
            z = z.vector()

        x_all = [x, m, y, z]
        pde[n_pde].solveFwd(x, x_all, tol)
        Q_0 = qoi[n_pde].eval(x) # Q(x(m_0))
        x_all[STATE] = x
        # solve the adjoint problem at the mean
        rhs = pde[n_pde].generate_state()
        qoi[n_pde].adj_rhs(x,rhs)
        pde[n_pde].solveAdj(y,x_all,rhs, tol)
        x_all[ADJOINT] = y

        # set the linearization point for incremental forward and adjoint problems
        pde[n_pde].setLinearizationPoint(x_all)
        qoi[n_pde].setLinearizationPoint(x)

        ######### randomized SVD estimator #########
        Hessian = ReducedHessianSVD(pde[n_pde], qoi[n_pde], tol)

        randomGen = Random(myid=0, nproc=mpi_size)
        Omega = MultiVector(m, N_tr+50)
        for i in xrange(N_tr+50):
            randomGen.normal(1., Omega[i])

        d, U = doublePassG(Hessian, prior.R, prior.Rsolver, Omega, N_tr+40)
                                     # check_Bortho = True, check_Aortho=False, check_residual = True)
        traceTrue = 0.5*sum(d)

        d = d[0:N_tr-1]

        ##### randomized MC estimator #########
        noise = dl.Vector()
        sample = dl.Vector()
        prior.init_vector(noise, "noise")
        prior.init_vector(sample, 1)
        noise_size = noise.get_local().shape[0]

        m_tr = dict()
        for i in range(N_tr):
            sample = dl.Vector()
            noise.set_local(np.random.normal(0, 1, noise_size))
            prior.sample(noise, sample, add_mean=False)
            m_tr[i] = sample

        dmmr = np.zeros(N_tr)
        for i in range(N_tr):
            mhat = m_tr[i]
            dmmr[i], xhat, yhat = Hessian.HessianInner(mhat, mhat)

        #############error computation ####################
        error_mc = np.zeros(N_tr)
        error_svd = np.zeros(N_tr)
        for i in range(1, N_tr):
            error_mc[i] = np.abs((traceTrue-0.5*np.mean(dmmr[0:i])))
            error_svd[i] = np.abs((traceTrue-0.5*np.sum(d[0:i])))

        if mpi_rank == 0:
            if test == 0:
                np.savez("data/traceMCvsSVDrandomPDE"+str(n_pde)+".npz", d=d, error_mc=error_mc, error_svd=error_svd)
            elif test == 1:
                np.savez("data/traceMCvsSVDconstantPDE"+str(n_pde)+".npz", d=d, error_mc=error_mc, error_svd=error_svd)
            elif test == 2:
                np.savez("data/traceMCvsSVDlinearPDE"+str(n_pde)+".npz", d=d, error_mc=error_mc, error_svd=error_svd)
            elif test == 3:
                np.savez("data/traceMCvsSVDquadraticPDE"+str(n_pde)+".npz", d=d, error_mc=error_mc, error_svd=error_svd)
            elif test == 4:
                np.savez("data/traceMCvsSVDsaaPDE"+str(n_pde)+".npz", d=d, error_mc=error_mc, error_svd=error_svd)

        if plotFlag:
            fig = plt.figure()
            indexplus = np.where(d > 0)[0]
            # print indexplus
            dplus, = plt.semilogy(indexplus, d[indexplus], 'ro')
            indexminus = np.where(d < 0)[0]
            # print indexminus
            dminus, = plt.semilogy(indexminus, -d[indexminus], 'k*')
            # plt.draw()
            # plt.pause(0.001)
            # plt.clf()
            e_mc, = plt.semilogy(error_mc,'d')
            e_svd, = plt.semilogy(error_svd,'x')
            plt.xlabel("$N$",fontsize=20)
            plt.legend([dplus, dminus, e_mc, e_svd], ["$\lambda_+$", "$\lambda_-$", "errorMC", "errorSVD"], fontsize=20, loc=3)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tick_params(axis='both', which='minor', labelsize=16)

            if test == 0:
                fig.savefig("figure/traceMCvsSVDrandomPDE"+str(n_pde)+".eps",format='eps')
            elif test == 1:
                fig.savefig("figure/traceMCvsSVDconstantPDE"+str(n_pde)+".eps",format='eps')
            elif test == 2:
                fig.savefig("figure/traceMCvsSVDlinearPDE"+str(n_pde)+".eps",format='eps')
            elif test == 3:
                fig.savefig("figure/traceMCvsSVDquadraticPDE"+str(n_pde)+".eps",format='eps')
            elif test == 4:
                fig.savefig("figure/traceMCvsSVDsaaPDE"+str(n_pde)+".eps",format='eps')

if plotFlag:
    plt.show()

