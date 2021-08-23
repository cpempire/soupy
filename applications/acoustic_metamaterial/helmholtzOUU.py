### optimization under uncertainty constrained by Helmholtz model

from construction import *
from shutil import copyfile

################### 6. save data ##########################################
def initialdata(z):

    cost.costValue(z)

    for i in range(N_pde):
        x = cost.x[i]
        x_fun = vector2Function(x, Vh_STATE, name="state")
        ur, ui = x_fun.split()

        if plotFlag:
            dl.plot(ur, title='init state real')
            dl.plot(ui, title='init state imag')

        if dlversion() <= (1, 6, 0):
            dl.File(comm, "data/ur_init_"+str(i)+".xdmf") << ur
            dl.File(comm, "data/ui_init_"+str(i)+".xdmf") << ui
        else:
            xf = dl.XDMFFile(comm, "data/ur_init_"+str(i)+".xdmf")
            xf.write(ur)
            xf = dl.XDMFFile(comm, "data/ui_init_"+str(i)+".xdmf")
            xf.write(ui)

    noise = dl.Vector()
    sample = dl.Vector()
    prior.init_vector(noise, "noise")
    prior.init_vector(sample, 1)
    noise_size = noise.get_local().shape[0]

    for i in range(4):
        sample = dl.Vector()
        noise.set_local(np.random.normal(0, 1, noise_size))
        prior.sample(noise, sample, add_mean=True)
        sample_fun = vector2Function(sample, Vh[PARAMETER], name="sample")
        sample_fun = dl.project(sample_fun * chi_design, Vh_PARAMETER)
        sample_fun = vector2Function(sample_fun.vector(), Vh[PARAMETER], name="sample")
        if plotFlag:
            dl.plot(sample_fun)

        # if mpi_size == 1:
        #     filename = 'data/sample_'+str(i)+'.pvd'
        #     dl.File(filename) << sample_fun
        # else:
        filename = 'data/sample_' + str(i) + '.xdmf'
        if dlversion() <= (1, 6, 0):
            dl.File(comm, filename) << sample_fun
        else:
            xf = dl.XDMFFile(comm, filename)
            xf.write(sample_fun)

def savedata(type):

    for i in range(N_pde):
        x = cost.x[i]
        if type is 'saa':
            x = x[0]
        x_fun = vector2Function(x, Vh_STATE, name="state")

        ur, ui = x_fun.split()

        if plotFlag:
            dl.plot(ur, title='optimal state real')
            dl.plot(ui, title='optimal state imag')
            dl.plot(z_fun, title="optimal control")

        if dlversion() <= (1, 6, 0):
            dl.File(comm, "data/" + type + "/ur_opt_"+str(i)+".xdmf") << ur
            dl.File(comm, "data/" + type + "/ui_opt_"+str(i)+".xdmf") << ui
        else:
            xf = dl.XDMFFile(comm, "data/" + type + "/ur_opt_"+str(i)+".xdmf")
            xf.write(ur)
            xf = dl.XDMFFile(comm, "data/" + type + "/ui_opt_"+str(i)+".xdmf")
            xf.write(ui)

    z_fun = vector2Function(cost.z, Vh_OPTIMIZATION, name="design")
    z1, z2, z3 = z_fun.split()
    z1, z2, z3 = dl.interpolate(z1, Vh_OPTIMIZATION_sub), dl.interpolate(z2, Vh_OPTIMIZATION_sub), dl.interpolate(z3, Vh_OPTIMIZATION_sub)
    output_file = dl.HDF5File(comm, "data/" + type + "/z_opt_load.h5", "w")
    output_file.write(z_fun, "z_opt")
    output_file.close()

    if dlversion() <= (1, 6, 0):
        dl.File(comm, "data/" + type + "/z1_opt.xdmf") << z1
        dl.File(comm, "data/" + type + "/z2_opt.xdmf") << z2
        dl.File(comm, "data/" + type + "/z3_opt.xdmf") << z3
    else:
        xf = dl.XDMFFile(comm, "data/" + type + "/z1_opt.xdmf")
        xf.write(z1)
        xf = dl.XDMFFile(comm, "data/" + type + "/z2_opt.xdmf")
        xf.write(z2)
        xf = dl.XDMFFile(comm, "data/" + type + "/z3_opt.xdmf")
        xf.write(z3)

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

# bounds = [(0,1.)]
# for i in range(z.shape[0]-1):
#     bounds.append((0,1.))

bounds = None
if optMethod is 'home_bfgs':
    maxiter = bfgs_ParameterList["max_iter"]
elif optMethod is 'home_ncg':
    maxiter = ncg_ParameterList["max_iter"]
else:
    maxiter = 32

def optimization(cost):

    z = pde[0].generate_optimization()
    if optMethod is not 'home_bfgs' and optMethod is not 'home_ncg':
        z = np.zeros(z.size())

    if check_gradient:
        cost.checkGradient(z, mpi_comm, plotFlag=False)
    if check_hessian:
        cost.checkHessian(z, mpi_comm, plotFlag=False)

    if optMethod is 'home_bfgs':
        bfgs_Solver = BFGS(cost, bfgs_ParameterList)
        h0inv = H0inv(penalization)
        z = pde[0].generate_optimization()
        opt_result = bfgs_Solver.solve(z, h0inv)
    elif optMethod is 'home_ncg':
        ncg_Solver = ReducedSpaceNewtonCG(cost, ncg_ParameterList)
        z = pde[0].generate_optimization()
        opt_result = ncg_Solver.solve(z)
    elif optMethod is 'sci_bfgs':
        opt_result = fmin_l_bfgs_b(func=cost.costValue, x0=z, fprime=cost.costGradient,
                        disp=1, pgtol=1e-5, maxiter=maxiter, factr=1e12, bounds = bounds)
    elif optMethod is 'fmin_ncg':
        opt_result = fmin_ncg(f=cost.costValue, x0=z, fprime=cost.costGradient, fhess_p=cost.costHessian,
                        avextol=1e-04, maxiter=maxiter, full_output=1, disp=1, retall=1)

    return cost, opt_result

if constant_run:
    ######## Zero-th approximation ########################
    cost = CostFunctionalConstantMultiPDE(parameter, Vh, pde, qoi, prior, penalization, tol=1e-10)

    # generate and save initial data
    z = pde[0].generate_optimization()
    if optMethod is not 'home_bfgs' and optMethod is not 'home_ncg':
        z = np.zeros(z.size())

    initialdata(z)

    cost, opt_result = optimization(cost)

    savedata("constant")

if linear_run:
    ######## Linear approximation ########################
    cost = CostFunctionalLinearMultiPDE(parameter, Vh, pde, qoi, prior, penalization, tol=1e-10)

    cost, opt_result = optimization(cost)

    savedata("linear")

if quadratic_run:
    ######### Quadratic approximation ########################
    cost = CostFunctionalQuadraticMultiPDE(parameter, Vh, pde, qoi, prior, penalization, tol=1e-10)

    cost, opt_result = optimization(cost)

    savedata("quadratic")

if saa_run:
    ######## sample average approximation ########################
    cost = CostFunctionalSAAMultiPDE(parameter, Vh, pde, qoi, prior, penalization, tol=1e-10)

    cost, opt_result = optimization(cost)

    savedata("saa")

if plotFlag:
    dl.interactive()