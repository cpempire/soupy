from __future__ import absolute_import, division, print_function

plotFlag = True

if plotFlag:
    import matplotlib.pyplot as plt

import numpy as np
import pickle

type_set = ['constant', 'linear', 'quadratic', 'saa']

size = 1

meshsize_set = [1, 2, 3, 4, 5]
mesh = ['dim = 3,887', 'dim = 14,991', 'dim = 58,956', 'dim = 233,311', 'dim = 929,763']

fig = plt.figure()
h = []
for type in type_set:

    filename = "data/"+type+"/data.p"
    data = pickle.load(open(filename, 'rb'))
    iteration = data['opt_result']['Iteration']
    costValue = data['opt_result']['costValue']
    costGrad = data['opt_result']['costGrad']

    if type is 'constant':
        h1, = plt.semilogy(iteration, costValue, 'b.-')
        h.append(h1)
    elif type is 'linear':
        h2, = plt.semilogy(iteration, costValue, 'rx-')
        h.append(h2)
    elif type is 'quadratic':
        h3, = plt.semilogy(iteration, costValue, 'gd-')
        h.append(h3)
    elif type is 'saa':
        h4, = plt.semilogy(iteration, costValue, 'ks-')
        h.append(h4)


plt.xlabel("# BFGS iterations",fontsize=16)
plt.ylabel("cost",fontsize=16)

plt.legend(h, type_set, fontsize=12, loc=1)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

filename = "figure/BFGS.eps"
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
        filename = "run_disk"+str(meshsize)+"_correction"+"/data/"+type+"/data_l1.p"
        data = pickle.load(open(filename,'rb'))
        iteration = data['opt_result']['Iteration']
        costValue = data['opt_result']['costValue']
        costGrad = data['opt_result']['costGrad']

        if meshsize == 1:
            h1, = plt.semilogy(iteration, costValue, 'b.-')
            h.append(h1)
        elif meshsize == 2:
            h2, = plt.semilogy(iteration, costValue, 'rx-')
            h.append(h2)
        elif meshsize == 3:
            h3, = plt.semilogy(iteration, costValue, 'gd-')
            h.append(h3)
        elif meshsize == 4:
            h4, = plt.semilogy(iteration, costValue, 'ks-')
            h.append(h4)
        elif meshsize == 5:
            h5, = plt.semilogy(iteration, costValue, 'm<-')
            h.append(h5)

    if type is 'random':
        title = "at random design"
    else:
        title = "at optimal design with " + type +" approximation"
    plt.title(title, fontsize=16)
    plt.xlabel("# BFGS iterations",fontsize=16)
    plt.ylabel("cost",fontsize=16)
    plt.legend(h[0:size], mesh[0:size], fontsize=20, loc=1)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    filename = "figure/BFGS"+type+"_correction"+".eps"
    fig.savefig(filename,format='eps')

if plotFlag:
    plt.show()