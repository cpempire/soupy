### optimization under uncertainty

from construction import *
from shutil import copyfile

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
                        disp=1, pgtol=1e-3, maxiter=maxiter, factr=1e7, bounds=cost.parameter["bounds"], iprint=99)
    elif optMethod is 'fmin_ncg':
        opt_result = fmin_ncg(f=cost.costValue, x0=z, fprime=cost.costGradient, fhess_p=cost.costHessian,
                        avextol=1e-04, maxiter=maxiter, full_output=1, disp=1, retall=1)

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

quadratic_run = True
if quadratic_run:
    ######### Quadratic approximation ########################
    cost = ChanceConstraintQuadratic(parameter, Vh, pde, qoi_objective, qoi_constraint, prior, penalization, tol=1e-10)

    cost.costValue(z)
    cost.errorAnalysis(0)

    cost, opt_result = optimization(cost, z)

    cost.errorAnalysis(1)

    c_beta = parameter["c_beta"]
    c_gamma = parameter["c_gamma"]

    print("update parameters c_beta = ", c_beta, " and c_gamma = ", c_gamma)
    print("optimization result", opt_result)
    print("func_ncalls, grad_ncalls, hess_calls = ", cost.func_ncalls, cost.grad_ncalls, cost.hess_ncalls)

    for i in range(num_iteration):
        c_beta *= factor_beta
        c_gamma *= factor_gamma
        cost.update(c_beta, c_gamma)
        z = cost.z.gather_on_zero()
        if mpi_size > 1:
            z = comm_mpi4py.bcast(z, root=0)
        cost.niters = 0
        cost, opt_result = optimization(cost, z)

        cost.errorAnalysis(i+2)

        print("update parameters c_beta = ", c_beta, " and c_gamma = ", c_gamma)
        print("optimization result", opt_result)
        print("func_ncalls, grad_ncalls, hess_calls = ", cost.func_ncalls, cost.grad_ncalls, cost.hess_ncalls)
