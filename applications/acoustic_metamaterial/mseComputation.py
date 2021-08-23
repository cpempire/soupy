# compute mean square error

from construction import *

if plotFlag:
    import matplotlib.pyplot as plt

noise = dl.Vector()
sample = dl.Vector()
prior.init_vector(noise, "noise")
prior.init_vector(sample, 1)
noise_size = noise.get_local().shape[0]

def mseRun(type):

    z = pde[0].generate_optimization()
    if optMethod is not 'home_bfgs' and optMethod is not 'home_ncg':
        z = np.zeros(z.size())

    if type is 'random':
        if mpi_rank == 0:
            print("="*10+"at random control"+"="*10)
        if isinstance(z, np.ndarray):
            z = np.random.normal(0, 0.1, z.shape[0])
        else:
            z.set_local(np.random.normal(0, 0.1, z.get_local().shape[0]))
            z_fun = vector2Function(z, Vh_OPTIMIZATION)
            z_fun = dl.project(z_fun*chi_design, Vh_OPTIMIZATION)
            if dlversion() <= (1,6,0):
                dl.File(mesh.mpi_comm(), "data/z_random.xdmf") << z_fun
            else:
                xf = dl.XDMFFile(mesh.mpi_comm(), "data/z_random.xdmf")
                xf.write(z_fun)

    else:
        if mpi_rank == 0:
            print("="*10+"at optimal control with "+type+" approximation"+"="*10)

        z = dl.Function(Vh_OPTIMIZATION)
        input_file = dl.HDF5File(mesh.mpi_comm(), "data/"+type+"/z_opt_load.h5",'r')
        input_file.read(z, 'z_opt')
        input_file.close()
        z = z.vector()

        if isinstance(z, np.ndarray):
            z = z.gather_on_zero()
            if mpi_size > 1:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                z = comm.bcast(z, root=0)

    cost.costValue(z)

    for n_pde in range(N_pde):
        Q_mean = np.zeros(N_mse)
        lin_diff_mean= np.zeros(N_mse)
        quad_diff_mean= np.zeros(N_mse)
        Q_var = np.zeros(N_mse)
        lin_diff_var = np.zeros(N_mse)
        quad_diff_var = np.zeros(N_mse)

        u_mean = dl.Function(Vh_STATE).vector()
        u_var = dl.Function(Vh_STATE).vector()
        for i in range(N_mse):
            sample = dl.Vector()
            noise.set_local(np.random.normal(0, 1, noise_size))
            prior.sample(noise, sample, add_mean=False)

            u, Q_mean[i], lin_diff_mean[i], quad_diff_mean[i], Q_var[i], lin_diff_var[i], quad_diff_var[i] \
                = cost.QuadraticSampling(sample, savedata=False, i=i, type=type, n_pde=n_pde)
            u_mean.axpy(1.0, u)
            u[:] = u[:]*u[:]
            u_var.axpy(1.0, u)

            if mpi_rank == 0:
                header = ['i-th', 'mean', 'lin_diff', 'quad_diff']
                print('{:<20} {:<20} {:<20} {:<20}'.format(*header))
                data = [i,  Q_mean[i], lin_diff_mean[i], quad_diff_mean[i]]
                print('{:<20} {:<20.2e} {:<20.2e} {:<20.2e}'.format(*data))
                header = ['i-th', 'var', 'lin_diff', 'quad_diff']
                print('{:<20} {:<20} {:<20} {:<20}'.format(*header))
                data = [i,  Q_var[i], lin_diff_var[i], quad_diff_var[i]]
                print('{:<20} {:<20.2e} {:<20.2e} {:<20.2e}'.format(*data))

        u_mean[:] = u_mean[:]/N_mse
        u_var[:] = u_var[:]/N_mse - u_mean[:]*u_mean[:]

        umean = vector2Function(u_mean, Vh_STATE)
        uvar =  vector2Function(u_var, Vh_STATE)

        umean = dl.project(umean*chi_observation, Vh_STATE)
        umean = vector2Function(umean.vector(), Vh_STATE, name='mean')
        urmean, uimean = umean.split()

        filename_ur = "data/"+"ur_"+type+'_mean'+'_PDE_'+str(n_pde)+'.xdmf'
        filename_ui = "data/"+"ui_"+type+'_mean'+'_PDE_'+str(n_pde)+'.xdmf'
        if dlversion() <= (1, 6, 0):
            dl.File(mesh.mpi_comm(), filename_ur) << urmean
            dl.File(mesh.mpi_comm(), filename_ui) << uimean
        else:
            xf = dl.XDMFFile(mesh.mpi_comm(), filename_ur)
            xf.write(urmean)
            xf = dl.XDMFFile(mesh.mpi_comm(), filename_ui)
            xf.write(uimean)

        uvar = dl.project(uvar*chi_observation, Vh_STATE)
        uvar = vector2Function(uvar.vector(), Vh_STATE, name='std')
        urvar, uivar = uvar.split()

        filename_ur = "data/"+"ur_"+type+'_var'+'_PDE_'+str(n_pde)+'.xdmf'
        filename_ui = "data/"+"ui_"+type+'_var'+'_PDE_'+str(n_pde)+'.xdmf'
        if dlversion() <= (1, 6, 0):
            dl.File(mesh.mpi_comm(), filename_ur) << urvar
            dl.File(mesh.mpi_comm(), filename_ui) << uivar
        else:
            xf = dl.XDMFFile(mesh.mpi_comm(), filename_ur)
            xf.write(urvar)
            xf = dl.XDMFFile(mesh.mpi_comm(), filename_ui)
            xf.write(uivar)

        if mpi_rank == 0:
            np.savez("data/mseResults_"+type+'_PDE_'+str(n_pde)+".npz", Q_mean=Q_mean,lin_diff_mean=lin_diff_mean,quad_diff_mean=quad_diff_mean,
                                                Q_var=Q_var,lin_diff_var=lin_diff_var,quad_diff_var=quad_diff_var)
            step = N_mse/10

            header = ['# samples', 'Q_mean', 'Q_mean_mse','lin_diff_mean_mse','quad_diff_mean_mse']
            print('{:<20} {:<20} {:<20} {:<20} {:<20}'.format(*header))
            for i in range(10):
                data = [(i+1)*step, np.mean(Q_mean[0:(i+1)*step:1]),
                        np.var(Q_mean[0:(i+1)*step:1])/((i+1)*step),
                        np.var(lin_diff_mean[0:(i+1)*step:1])/((i+1)*step),
                        np.var(quad_diff_mean[0:(i+1)*step:1])/((i+1)*step)]
                print('{:<20.2e} {:<20.2e} {:<20.2e} {:<20.2e} {:<20.2e}'.format(*data))

            header = ['# samples', 'Q_var', 'Q_var_mse','lin_diff_var_mse','quad_diff_var_mse']
            print('{:<20} {:<20} {:<20} {:<20} {:<20}'.format(*header))
            for i in range(10):
                data = [(i+1)*step, np.mean(Q_var[0:(i+1)*step:1]),
                        np.var(Q_var[0:(i+1)*step:1])/((i+1)*step),
                        np.var(lin_diff_var[0:(i+1)*step:1])/((i+1)*step),
                        np.var(quad_diff_var[0:(i+1)*step:1])/((i+1)*step)]
                print('{:<20} {:<20.2e} {:<20.2e} {:<20.2e} {:<20.2e}'.format(*data))

        if plotFlag:
            plt.figure()
            lin_diff_mean_plot, = plt.semilogy(np.abs(lin_diff_mean/Q_mean), 'ko')
            quad_diff_mean_plot, = plt.semilogy(np.abs(quad_diff_mean/Q_mean), 'rx')
            plt.legend([lin_diff_mean_plot,quad_diff_mean_plot], ['linear mean error', 'quad mean error'])

            plt.figure()
            lin_diff_var_plot, = plt.semilogy(np.abs(lin_diff_var/Q_var), 'ko')
            quad_diff_var_plot, = plt.semilogy(np.abs(quad_diff_var/Q_var), 'rx')
            plt.legend([lin_diff_var_plot,quad_diff_var_plot], ['linear var error', 'quad var error'])

################### 6. construct a quadratic approximation #################

parameter['N_mc'] = 1
parameter['N_tr'] = 1

cost = CostFunctionalQuadraticMultiPDE(parameter, Vh, pde, qoi, prior, penalization, tol=1e-8)

for type in ['random', 'constant', 'linear', 'quadratic', 'saa']:
    mseRun(type)

if plotFlag:
    plt.show()