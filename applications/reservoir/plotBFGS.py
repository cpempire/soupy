from __future__ import absolute_import, division, print_function

plotFlag = True

if plotFlag:
    import matplotlib.pyplot as plt

plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

meshsize_set = [1, 2, 3, 4, 5]
mesh = ["dim = 1,089", "dim = 4,225", "dim = 16,641", "dim = 66,049", "dim = 263,169"]
outloop = ["step 1", "step 2", "step 3", "step 4", "step 5"]
marker = ["b.-", "r.-", "g.-", "k.-", "m.-"]

fig = plt.figure(1)
data = []
for size in meshsize_set:

    filename = "mesh"+str(size)+"/iterate.dat"
    file = open(filename, 'r')

    lines = file.readlines()
    count = -10000
    data_line = []
    data_count = []
    fig_line = plt.figure(2)
    qiter = 0
    for line in lines:
        if " quadratic approximation" in line:
            count = 0
            if len(data_count) > 0 and size == 1:
                data_line.append(data_count)
                plt.figure(2)
                res = next(x[0] for x in enumerate(data_count) if x[1] < 1e-3)
                res = min(res, len(data_count)-1)
                plt.semilogy(data_count[:res+1], marker[qiter], label=outloop[qiter])
                qiter += 1

            data_count = []
        count += 1
        if count >= 3:
            column = line.split()
            # print("column", column, float(column[3]))
            data_count.append(float(column[3]))

    if len(data_count) > 0 and size == 1:
        data_line.append(data_count)
        plt.figure(2)
        res = next(x[0] for x in enumerate(data_count) if x[1] < 1e-3)
        res = min(res, len(data_count) - 1)
        plt.semilogy(data_count[:res+1], marker[qiter], label=outloop[qiter])
        qiter += 1

    plt.figure(2)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.legend(loc="best", fontsize=12)
    plt.xlabel("iteration", fontsize=16)
    plt.ylabel("gradient norm", fontsize=16)
    filename = "figure/BFGS" + str(size) + ".pdf"
    plt.savefig(filename)
    plt.close()

    data.append(data_line)

    plt.figure(1)
    res = next(x[0] for x in enumerate(data_line[-1]) if x[1] < 1e-3)
    res = min(res, len(data_line[-1]) - 1)
    plt.semilogy(data_line[-1][:res+1], marker[size-1], label=mesh[size-1])

plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.legend(loc="best", fontsize=12)
plt.xlabel("iteration", fontsize=16)
plt.ylabel("gradient norm", fontsize=16)
plt.savefig("figure/BFGS.pdf")

plt.close()