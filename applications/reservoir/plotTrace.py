from __future__ import absolute_import, division, print_function

plotFlag = True

if plotFlag:
    import matplotlib.pyplot as plt

import numpy as np

type_set = ['random', 'constant', 'linear', 'quadratic', 'saa']
mesh = ['dim = 3,887', 'dim = 14,991', 'dim = 58,956', 'dim = 233,311', 'dim = 929,763']

fig = plt.figure()
h = []

for type in type_set:

    filename = "data/traceMCvsSVD"+type+".npz"
    data = np.load(filename)
    d=data["d"]

    if type is 'random':
        h1, = plt.semilogy(np.abs(d), 'b.-')
        h.append(h1)
    elif type is 'constant':
        h2, = plt.semilogy(np.abs(d), 'rx-')
        h.append(h2)
    elif type is 'linear':
        h3, = plt.semilogy(np.abs(d), 'gd-')
        h.append(h3)
    elif type is 'quadratic':
        h4, = plt.semilogy(np.abs(d), 'ks-')
        h.append(h4)
    elif type is 'saa':
        h5, = plt.semilogy(np.abs(d), 'm<-')
        h.append(h5)

plt.xlabel("$N$",fontsize=16)
plt.ylabel("|$\lambda_N$|",fontsize=16)

plt.legend(h, type_set, fontsize=20, loc=1)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

filename = "figure/Eigenvalue.eps"
fig.savefig(filename,format='eps')

if plotFlag:
    plt.show()


# correction


type_set = ['linear','quadratic']
size = 3
meshsize_set = [1, 2, 3, 4, 5]
mesh = ['dim = 940', 'dim = 3,336', 'dim = 12,487', 'dim = 48,288', 'dim = 189,736'] #["5,809", "20,097", "79,873", "31,8465", "1,271,809"]

for type in type_set:

    fig = plt.figure()
    h = []
    for meshsize in meshsize_set[0:size]:
        filename = "run_disk"+str(meshsize)+"_correction"+"/data/traceMCvsSVD"+type+".npz"
        data = np.load(filename)
        d=data["d"]

        if meshsize == 1:
            h1, = plt.semilogy(np.abs(d), 'b.-')
            h.append(h1)
        elif meshsize == 2:
            h2, = plt.semilogy(np.abs(d), 'rx-')
            h.append(h2)
        elif meshsize == 3:
            h3, = plt.semilogy(np.abs(d), 'gd-')
            h.append(h3)
        elif meshsize == 4:
            h4, = plt.semilogy(np.abs(d), 'ks-')
            h.append(h4)
        elif meshsize == 5:
            h5, = plt.semilogy(np.abs(d), 'm<-')
            h.append(h5)

    if type is 'random':
        title = "at random design"
    else:
        title = "at optimal design with " + type +" approximation"
    plt.title(title, fontsize=16)
    plt.xlabel("$N$",fontsize=16)
    plt.ylabel("|$\lambda_N$|",fontsize=16)

    plt.legend(h[0:size], mesh[0:size], fontsize=20, loc=1)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    filename = "figure/Eigenvalue"+type+"_correction"+".eps"
    fig.savefig(filename,format='eps')

if plotFlag:
    plt.show()

# d_init = []
# d_opt = []
# for mesh in range(1, 6):
#     filemesh = "run_mesh"+str(mesh)
#     for test in range(2):
#         if test == 0:
#             filename = filemesh + "/data/traceMCvsSVDinit.npz"
#             data = np.load(filename)
#             d=data["d"]
#             d_init.append(d)
#         else:
#             filename = filemesh + "/data/traceMCvsSVDopt.npz"
#             data = np.load(filename)
#             d=data["d"]
#             d_opt.append(d)
#
#         error_mc=data["error_mc"]
#         error_svd=data["error_svd"]
#
#         if plotFlag:
#             fig = plt.figure()
#             indexplus = np.where(d > 0)[0]
#             # print indexplus
#             dplus, = plt.semilogy(indexplus, d[indexplus], 'ro')
#             indexminus = np.where(d < 0)[0]
#             # print indexminus
#             dminus, = plt.semilogy(indexminus, -d[indexminus], 'k*')
#             # plt.draw()
#             # plt.pause(0.001)
#             # plt.clf()
#             e_mc, = plt.semilogy(error_mc,'d')
#             e_svd, = plt.semilogy(error_svd,'x')
#             plt.xlabel("$N$",fontsize=20)
#             plt.legend([dplus, dminus, e_mc, e_svd], ["$\lambda_+$", "$\lambda_-$", "errorMC", "errorSVD"], fontsize=20, loc=3) #1,271,809
#             plt.tick_params(axis='both', which='major', labelsize=16)
#             plt.tick_params(axis='both', which='minor', labelsize=16)
#
#             if test == 0:
#                 filename = filemesh + "/figure/traceMCvsSVDinit.eps"
#                 fig.savefig(filename,format='eps')
#             elif test == 1:
#                 filename = filemesh + "/figure/traceMCvsSVDopt.eps"
#                 fig.savefig(filename,format='eps')
#
# plt.show()
#
# fig = plt.figure()
# d = d_init[0]
# d1, = plt.semilogy(d, 'b.')
# d = d_init[1]
# d2, = plt.semilogy(d, 'ro')
# d = d_init[2]
# d3, = plt.semilogy(d, 'k*')
# d = d_init[3]
# d4, = plt.semilogy(d, 'md')
# d = d_opt[4]
# d5, = plt.semilogy(d, 'cs')
# plt.xlabel("$N$",fontsize=20)
# plt.legend([d1,d2,d3,d4,d5], ["5,809", "20,097", "79,873", "31,8465","1,271,809"], fontsize=20, loc=3)
# plt.tick_params(axis='both', which='major', labelsize=16)
# plt.tick_params(axis='both', which='minor', labelsize=16)
# fig.savefig("figure/traceSVinit.eps",format='eps')
#
#
# fig = plt.figure()
# d = d_opt[0]
# d1, = plt.semilogy(d, 'b.')
# d = d_opt[1]
# d2, = plt.semilogy(d, 'ro')
# d = d_opt[2]
# d3, = plt.semilogy(d, 'k*')
# d = d_opt[3]
# d4, = plt.semilogy(d, 'md')
# d = d_opt[4]
# d5, = plt.semilogy(d, 'cs')
# plt.xlabel("$N$",fontsize=20)
# plt.legend([d1,d2,d3,d4,d5], ["5,809", "20,097", "79,873", "31,8465","1,271,809"], fontsize=20, loc=3)
# plt.tick_params(axis='both', which='major', labelsize=16)
# plt.tick_params(axis='both', which='minor', labelsize=16)
# fig.savefig("figure/traceSVopt.eps",format='eps')
# plt.show()