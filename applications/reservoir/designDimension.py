import dolfin as dl
import numpy as np
path = "../"

geometry = 'square'


chi_design = dl.Expression("((pow(x[0]-6.,2)+pow(x[1]-6.,2))>=1)*((pow(x[0]-6.,2)+pow(x[1]-6.,2))<=9)", degree=1)

meshsize_set = [1, 2, 3, 4, 5]

dim_z = []
dim_m = []
dim_u = []
for meshsize in meshsize_set:

    filename = path+"mesh/"+geometry+str(meshsize)+".xml"
    mesh = dl.Mesh(filename)
    Vh_STATE = dl.VectorFunctionSpace(mesh, 'CG', 1)
    Vh_PARAMETER = dl.FunctionSpace(mesh, 'CG', 1)
    Vh_OPTIMIZATION = dl.FunctionSpace(mesh, "DG", 0)
    chi_design = dl.interpolate(chi_design, Vh_OPTIMIZATION)

    z_test = dl.TestFunction(Vh_OPTIMIZATION)
    z = dl.assemble(z_test*chi_design*dl.dx)
    m_test = dl.TestFunction(Vh_PARAMETER)
    m = dl.assemble(m_test * chi_design * dl.dx)
    u = dl.Function(Vh_STATE)

    dim_z.append(np.count_nonzero(z.get_local()))
    dim_m.append(np.count_nonzero(m.get_local()))
    dim_u.append(u.vector().size())

print("dim_z", dim_z)
print("dim_m", dim_m)
print("dim_u", dim_u)