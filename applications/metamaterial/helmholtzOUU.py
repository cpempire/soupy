### optimization under uncertainty constrained by Helmholtz model

from construction import *
from shutil import copyfile

################### 6. save data ##########################################
def initialdata(z):

    cost.costValue(z)

    x = cost.x
    x_fun = vector2Function(x, Vh_STATE, name="state")
    ur_scattered, ui_scattered = x_fun.split()

    if plotFlag:
        dl.plot(ur_scattered, title='init state real')
        dl.plot(ui_scattered, title='init state imag')

    # if mpi_size == 1:
    #     dl.File("data/constant/ur_init.pvd") << ur_scattered
    #     dl.File("data/constant/ui_init.pvd") << ui_scattered
    # else:

    u_total = dl.Function(Vh_STATE, name="state")
    ur_total, ui_total = u_total.split()
    ur_total, ui_total = dl.interpolate(ur_total, Vhsub), dl.interpolate(ui_total, Vhsub)
    ur_scattered, ui_scattered = dl.interpolate(ur_scattered, Vhsub), dl.interpolate(ui_scattered, Vhsub)
    ur_total.vector().axpy(1.0, ur_scattered.vector())
    ur_total.vector().axpy(1.0, ur_incident.vector())
    ui_total.vector().axpy(1.0, ui_scattered.vector())
    ui_total.vector().axpy(1.0, ui_incident.vector())

    if dlversion() <= (1,6,0):
        dl.File(comm, "data/ur_init_scattered.xdmf") << ur_scattered
        dl.File(comm, "data/ui_init_scattered.xdmf") << ui_scattered
        dl.File(comm, "data/ur_init_total.xdmf") << ur_total
        dl.File(comm, "data/ui_init_total.xdmf") << ui_total
    else:
        xf = dl.XDMFFile(comm, "data/ur_init_scattered.xdmf")
        xf.write(ur_scattered)
        xf = dl.XDMFFile(comm, "data/ui_init_scattered.xdmf")
        xf.write(ui_scattered)
        xf = dl.XDMFFile(comm, "data/ur_init_total.xdmf")
        xf.write(ur_total)
        xf = dl.XDMFFile(comm, "data/ui_init_total.xdmf")
        xf.write(ui_total)

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
        chi = dl.Expression("((pow(x[0]-10.,2)+pow(x[1]-5.,2))>=4)*((pow(x[0]-10.,2)+pow(x[1]-5.,2))<=16)", degree=2)
        sample_fun = dl.project(sample_fun*chi, Vh_PARAMETER)
        sample_fun = vector2Function(sample_fun.vector(),Vh[PARAMETER], name="sample")
        if plotFlag:
            dl.plot(sample_fun)

        # if mpi_size == 1:
        #     filename = 'data/sample_'+str(i)+'.pvd'
        #     dl.File(filename) << sample_fun
        # else:
        filename = 'data/sample_'+str(i)+'.xdmf'
        if dlversion() <= (1,6,0):
            dl.File(comm,filename) << sample_fun
        else:
            xf = dl.XDMFFile(comm, filename)
            xf.write(sample_fun)

def savedata(type):

    x_fun = vector2Function(cost.x, Vh_STATE, name="state")
    m_fun = vector2Function(cost.m, Vh_PARAMETER)
    z_fun = vector2Function(cost.z,Vh_OPTIMIZATION,name="design")

    ur_scattered, ui_scattered = x_fun.split()

    kopt = wavenumber(m_fun,z_fun,obs=False)
    kopt = dl.project(kopt,Vh_OPTIMIZATION)
    kopt = vector2Function(kopt.vector(),Vh_OPTIMIZATION,name="wavenumber")

    if plotFlag:
        dl.plot(ur_scattered, title = 'optimal state real')
        dl.plot(ui_scattered, title = 'optimal state imag')
        dl.plot(z_fun, title="optimal control")
        dl.plot(kopt, title="optimal wavenumber")

    # if mpi_size == 1:
    #     dl.File("data/"+type+"/ur_opt.pvd") << ur_scattered
    #     dl.File("data/"+type+"/ui_opt.pvd") << ui_scattered
    #     dl.File("data/"+type+"/z_opt.pvd") << z_fun
    #     dl.File("data/"+type+"/k_opt.pvd") << kopt
    # else:
    if dlversion() <= (1,6,0):
        dl.File(comm, "data/"+type+"/ur_opt_scattered.xdmf") << ur_scattered
        dl.File(comm, "data/"+type+"/ui_opt_scattered.xdmf") << ui_scattered
        dl.File(comm, "data/"+type+"/z_opt.xdmf") << z_fun
        dl.File(comm, "data/"+type+"/k_opt.xdmf") << kopt

    else:
        xf = dl.XDMFFile(comm, "data/"+type+"/ur_opt_scattered.xdmf")
        xf.write(ur_scattered)
        xf = dl.XDMFFile(comm, "data/"+type+"/ui_opt_scattered.xdmf")
        xf.write(ui_scattered)
        xf = dl.XDMFFile(comm, "data/"+type+"/z_opt.xdmf")
        xf.write(z_fun)
        xf = dl.XDMFFile(comm, "data/"+type+"/k_opt.xdmf")
        xf.write(kopt)

    output_file = dl.HDF5File(comm, "data/"+type+"/z_opt_load.h5", "w")
    output_file.write(z_fun, "z_opt")
    output_file.close()

    if mpi_rank == 0:
        print("optimization result", opt_result)
        print("func_ncalls, grad_ncalls = ", cost.func_ncalls, cost.grad_ncalls)

        data = dict()
        data['opt_result'] = opt_result
        data['tobj'] = cost.tobj
        data['tgrad'] = cost.tgrad
        data['trand'] = cost.trand
        data['func_ncalls'] = cost.func_ncalls
        data['grad_ncalls'] = cost.grad_ncalls

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

    z = pde.generate_optimization()
    if optMethod is not 'home_bfgs' and optMethod is not 'home_ncg':
        z = np.zeros(z.size())

    if check_gradient:
        cost.checkGradient(z, mpi_comm, plotFlag=False)
    if check_hessian:
        cost.checkHessian(z, mpi_comm, plotFlag=False)

    if optMethod is 'home_bfgs':
        bfgs_Solver = BFGS(cost, bfgs_ParameterList)
        h0inv = H0inv(penalization)
        z = pde.generate_optimization()
        opt_result = bfgs_Solver.solve(z, h0inv)
    elif optMethod is 'home_ncg':
        ncg_Solver = ReducedSpaceNewtonCG(cost, ncg_ParameterList)
        z = pde.generate_optimization()
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
    cost = CostFunctionalConstant(parameter, Vh, pde, qoi, prior, penalization, tol=1e-10)

    # generate and save initial data
    z = pde.generate_optimization()
    if optMethod is not 'home_bfgs' and optMethod is not 'home_ncg':
        z = np.zeros(z.size())

    initialdata(z)

    cost, opt_result = optimization(cost)

    savedata("constant")

if linear_run:
    ######## Linear approximation ########################
    cost = CostFunctionalLinear(parameter, Vh, pde, qoi, prior, penalization, tol=1e-10)

    cost, opt_result = optimization(cost)

    savedata("linear")

if quadratic_run:
    ######### Quadratic approximation ########################
    cost = CostFunctionalQuadratic(parameter, Vh, pde, qoi, prior, penalization, tol=1e-10)

    cost, opt_result = optimization(cost)

    savedata("quadratic")

if saa_run:
    ######## sample average approximation ########################
    cost = CostFunctionalSAA(parameter, Vh, pde, qoi, prior, penalization, tol=1e-10)

    cost, opt_result = optimization(cost)

    savedata("saa")

if plotFlag:
    dl.interactive()