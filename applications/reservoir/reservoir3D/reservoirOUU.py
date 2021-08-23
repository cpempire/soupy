### optimization under uncertainty

from construction import *
from shutil import copyfile

################### 6. save data ##########################################
def initialdata(z):

    cost.costValue(z)

    try:
        x = cost.x
        x_fun = vector2Function(x, Vh_STATE, name="state")

        if plotFlag:
            dl.plot(x_fun, title='init state')

        if dlversion() <= (1,6,0):
            dl.File(comm, "data/x_init.xdmf") << x_fun
        else:
            xf = dl.XDMFFile(comm, "data/x_init.xdmf")
            xf.write(x_fun)
            xf.close()

            xf = dl.HDF5File(comm, "data/x_init_hdf5.h5", "w")
            xf.write(x_fun, "x_init")
            xf.close()

    except:
        pass

    noise = dl.Vector()
    sample = dl.Vector()
    prior.init_vector(noise, "noise")
    prior.init_vector(sample, 1)
    noise_size = noise.get_local().shape[0]

    for i in range(4):
        sample = dl.Vector()
        noise.set_local(np.random.normal(0, 1, noise_size))
        prior.sample(noise, sample, add_mean=True)
        sample_fun = vector2Function(sample,Vh[PARAMETER], name="sample")
        if plotFlag:
            dl.plot(sample_fun)

        filename = 'data/sample_'+str(i)+'.xdmf'
        if dlversion() <= (1,6,0):
            dl.File(comm,filename) << sample_fun
        else:
            xf = dl.XDMFFile(comm, filename)
            xf.write(sample_fun)

def savedata(type):

    if type is 'saa':
        x_fun = vector2Function(cost.x[0], Vh_STATE, name="state")
        m_fun = vector2Function(cost.m[0], Vh_PARAMETER)
    else:
        x_fun = vector2Function(cost.x, Vh_STATE, name="state")
        m_fun = vector2Function(cost.m, Vh_PARAMETER)

    if plotFlag:
        dl.plot(x_fun, title='optimal state')

    if dlversion() <= (1,6,0):
        dl.File(comm, "data/"+type+"/x_opt.xdmf") << x_fun
    else:
        xf = dl.XDMFFile(comm, "data/"+type+"/x_opt.xdmf")
        xf.write(x_fun)
        xf.close()

        xf = dl.HDF5File(comm, "data/"+type+"/x_opt_hdf5.h5", "w")
        xf.write(x_fun, "x_opt")
        xf.close()
    # output_file = dl.HDF5File(comm, "data/"+type+"/z_opt_load.h5", "w")
    # output_file.write(z_fun, "z_opt")
    # output_file.close()

    if mpi_rank == 0:
        print("optimization result", opt_result)
        print("func_ncalls, grad_ncalls, hess_calls = ", cost.func_ncalls, cost.grad_ncalls, cost.hess_ncalls)

        data = dict()
        data['opt_result'] = opt_result
        data['tobj'] = cost.tobj
        data['tgrad'] = cost.tgrad
        data['trand'] = cost.trand
        data['func_ncalls'] = cost.func_ncalls
        data['grad_ncalls'] = cost.grad_ncalls
        data['hess_ncalls'] = cost.hess_ncalls

        data['lin_mean'] = cost.lin_mean
        data['lin_diff_mean'] = cost.lin_diff_mean
        data['lin_fval_mean'] = cost.lin_fval_mean
        data['lin_var'] = cost.lin_var
        data['lin_diff_var'] = cost.lin_diff_var
        data['lin_fval_var'] = cost.lin_fval_var

        data['quad_mean'] = cost.quad_mean
        data['quad_var'] = cost.quad_var
        data['quad_diff_mean'] = cost.quad_diff_mean
        data['quad_fval_mean'] = cost.quad_fval_mean
        data['quad_diff_var'] = cost.quad_diff_var
        data['quad_fval_var'] = cost.quad_fval_var

        pickle.dump( data, open( "data/"+type+"/data.p", "wb" ) )

        copyfile("iterate.dat", "data/"+type+"/iterate.dat")

################# 7. Solve the optimization algorithms #####################

if optMethod is 'home_bfgs':
    maxiter = bfgs_ParameterList["max_iter"]
elif optMethod is 'home_ncg':
    maxiter = ncg_ParameterList["max_iter"]
else:
    maxiter = 16

# bounds = [(0., 32.)]
# for i in range(15):
#     bounds.append((0., 32.))

# bounds = None
# bounds_min = pde.generate_optimization()
# bounds_min.zero()
# bounds_max = pde.generate_optimization()
# bounds_max.set_local(10.*np.ones(bounds_max.local_size()))
# bounds = [bounds_min, bounds_max]

def optimization(cost, z):

    if check_gradient:
        cost.checkGradient(z, mpi_comm, plotFlag=False)
    if check_hessian:
        cost.checkHessian(z, mpi_comm, plotFlag=False)

    if optMethod is 'home_bfgs':
        bfgs_Solver = BFGS(cost, bfgs_ParameterList)
        h0inv = H0inv(penalization)
        opt_result = bfgs_Solver.solve(z, h0inv, bounds_xPARAM=bounds)
    elif optMethod is 'home_ncg':
        ncg_Solver = ReducedSpaceNewtonCG(cost, ncg_ParameterList)
        opt_result = ncg_Solver.solve(z)
    elif optMethod is 'scipy_bfgs':
        opt_result = fmin_l_bfgs_b(func=cost.costValue, x0=z, fprime=cost.costGradient,
                        disp=1, pgtol=1e-4, maxiter=maxiter, factr=1e7, bounds=cost.parameter["bounds"], iprint=99,
                                   callback=cost.recording, maxls=10)
    elif optMethod is 'fmin_ncg':
        opt_result = fmin_ncg(f=cost.costValue, x0=z, fprime=cost.costGradient, fhess_p=cost.costHessian,
                        avextol=1e-03, maxiter=maxiter, full_output=1, disp=1, retall=1)

    return cost, opt_result


# generate and save initial data
z = pde.generate_optimization()
if optMethod is not 'home_bfgs' and optMethod is not 'home_ncg':
    z = 18. + np.zeros(z.size())

factor_beta = 2
factor_gamma = 10
num_iteration = 3

from mpi4py import MPI
comm_mpi4py = MPI.COMM_WORLD

if constant_run:
    ######## Zero-th approximation ########################
    cost = ChanceConstraintConstant(parameter, Vh, pde, qoi_objective, qoi_constraint, prior, penalization, tol=1e-10)

    initialdata(z)

    cost, opt_result = optimization(cost, z)

    c_beta = parameter["c_beta"]
    c_gamma = parameter["c_gamma"]
    for i in range(num_iteration):
        c_beta *= factor_beta
        c_gamma *= factor_gamma
        cost.update(c_beta, c_gamma)
        z = cost.z.gather_on_zero()
        if mpi_size > 1:
            z = comm_mpi4py.bcast(z, root=0)
        cost.niters = 0
        cost, opt_result = optimization(cost, z)

        print("update parameters c_beta = ", c_beta, " and c_gamma = ", c_gamma)
        print("optimization result", opt_result)
        print("func_ncalls, grad_ncalls, hess_calls = ", cost.func_ncalls, cost.grad_ncalls, cost.hess_ncalls)


    savedata("constant")

if linear_run:
    ######## Linear approximation ########################
    cost = ChanceConstraintLinear(parameter, Vh, pde, qoi_objective, qoi_constraint,  prior, penalization, tol=1e-10)

    cost, opt_result = optimization(cost, z)

    c_beta = parameter["c_beta"]
    c_gamma = parameter["c_gamma"]
    for i in range(num_iteration):
        c_beta *= factor_beta
        c_gamma *= factor_gamma
        cost.update(c_beta, c_gamma)
        z = cost.z.gather_on_zero()
        if mpi_size > 1:
            z = comm_mpi4py.bcast(z, root=0)
        cost.niters = 0
        cost, opt_result = optimization(cost, z)

        print("update parameters c_beta = ", c_beta, " and c_gamma = ", c_gamma)
        print("optimization result", opt_result)
        print("func_ncalls, grad_ncalls, hess_calls = ", cost.func_ncalls, cost.grad_ncalls, cost.hess_ncalls)

    savedata("linear")

if quadratic_run:
    ######### Quadratic approximation ########################
    cost = ChanceConstraintQuadratic(parameter, Vh, pde, qoi_objective, qoi_constraint, prior, penalization, tol=1e-10)

    cost, opt_result = optimization(cost, z)

    c_beta = parameter["c_beta"]
    c_gamma = parameter["c_gamma"]
    for i in range(num_iteration):
        c_beta *= factor_beta
        c_gamma *= factor_gamma
        cost.update(c_beta, c_gamma)
        z = cost.z.gather_on_zero()
        if mpi_size > 1:
            z = comm_mpi4py.bcast(z, root=0)
        cost.niters = 0
        cost, opt_result = optimization(cost, z)

        print("update parameters c_beta = ", c_beta, " and c_gamma = ", c_gamma)
        print("optimization result", opt_result)
        print("func_ncalls, grad_ncalls, hess_calls = ", cost.func_ncalls, cost.grad_ncalls, cost.hess_ncalls)

    savedata("quadratic")

if saa_run:
    ######## sample average approximation ########################
    cost = ChanceConstraintSAA(parameter, Vh, pde, qoi_objective, qoi_constraint, prior, penalization, tol=1e-10)

    cost, opt_result = optimization(cost, z)

    c_beta = parameter["c_beta"]
    c_gamma = parameter["c_gamma"]
    for i in range(num_iteration):
        c_beta *= factor_beta
        c_gamma *= factor_gamma
        cost.update(c_beta, c_gamma)
        z = cost.z.gather_on_zero()
        if mpi_size > 1:
            z = comm_mpi4py.bcast(z, root=0)
        cost.niters = 0
        cost, opt_result = optimization(cost, z)

        print("update parameters c_beta = ", c_beta, " and c_gamma = ", c_gamma)
        print("optimization result", opt_result)
        print("func_ncalls, grad_ncalls, hess_calls = ", cost.func_ncalls, cost.grad_ncalls, cost.hess_ncalls)

    savedata("saa")

if plotFlag:
    dl.interactive()