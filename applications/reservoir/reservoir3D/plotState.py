from construction import *
import matplotlib.pyplot as plt

x = [dl.Function(Vh[i]).vector() for i in range(4)]
state = dl.Function(Vh_STATE).vector()

filename = "data/quadratic/data.p"
data = pickle.load(open(filename, "r"))
z = 25.*np.ones(25) # data["opt_result"][0]
x[OPTIMIZATION].set_local(z)
x[PARAMETER].set_local(mtrue.get_local())

pde.solveFwd(state, x)

state_fun = dl.Function(Vh_STATE)
state_fun.vector().set_local(state.get_local())

velocity = -dl.grad(state_fun) # * dl.exp(mtrue_fun)
dl.plot(velocity)

x = np.linspace(0.25, 0.75, n1d)
y = np.linspace(0.25, 0.75, n1d)
xv,yv = np.meshgrid(x, y)
plt.plot(xv[:], yv[:], "bo")
plt.savefig("figure/x_opt_grad.pdf")

