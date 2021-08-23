import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)

marker = ["b.-", "r.-", "g.-", "k.-", "m.-"]
marker_objective = ["b.--", "r.--", "g.--", "k.--", "m.--"]
dim = ["step 0", "step 1", "step 2", "step 3", "step 4"]

for iter in range(5):

    filename = "data/errorAnalysis_" + str(iter) + ".p"
    data = pickle.load(open(filename, "r"))

    [f, f_0, f_taylor_0, f_taylor_1, f_taylor_2] = data["f"]
    [error_0, error_1, error_2] = data["error"]
    d_constraint = data["d_constraint"]
    # d_objective = data["d_objective"]
    # print("d_constraint", d_constraint, "d_objective", d_objective)

    mse_f = np.mean(np.square(f-np.mean(f)))/512
    mse_2 = np.mean(np.square(error_2))
    mse_1 = np.mean(np.square(error_1))
    mse_0 = np.mean(np.square(error_0))

    print("mse, 0, 1, 2", mse_f, mse_0, mse_1, mse_2)

    [l_beta_mean, l_beta_error, l_beta_0_mean, l_beta_1_mean, l_beta_2_mean] = data["accuracy"]

    print("l_beta_mean, l_beta_error, l_beta_0_mean, l_beta_1_mean, l_beta_2_mean = ", l_beta_mean,
          np.sqrt(l_beta_error), np.abs(l_beta_mean-l_beta_0_mean), np.abs(l_beta_mean-l_beta_1_mean), np.abs(l_beta_mean - l_beta_2_mean))

    print("plot eigenvalues")
    fig = plt.figure(10)
    plt.semilogy(np.abs(d_constraint), marker[iter], label=dim[iter])
    print("eigenvalues 10/100", np.sum(np.abs(d_constraint[:10]))/np.sum(np.abs(d_constraint)))
    # plt.semilogy(np.abs(d_objective), marker_objective[iter])
    plt.xlabel("$n$", fontsize=16)
    plt.ylabel("$|\lambda_n|$", fontsize=16)
    plt.legend(loc="best", fontsize=12)
    filename = "figure/Eigenvalue_" + str(iter) + ".pdf"
    fig.savefig(filename, format='pdf')
    # plt.close()

    print("plot Taylor approximation errors")
    fig = plt.figure()
    plt.semilogy(np.abs(f), 'b.', label='$|f|$')
    plt.semilogy(np.abs(error_0), 'gd', label="$|f - T_0 f|$")
    plt.semilogy(np.abs(error_1), 'rx', label="$|f - T_1 f|$")
    plt.semilogy(np.abs(error_2), 'ko', label='$|f - T_2 f|$')
    plt.legend(loc="best", fontsize=12)
    plt.xlabel("sample", fontsize=16)
    plt.ylabel("approximation errors", fontsize=16)
    filename = "figure/Taylor_" + str(iter) + ".pdf"
    fig.savefig(filename, format='pdf')
    plt.close()

    fig = plt.figure()
    # plt.hist(np.abs(f), 'b.', bins=50, label='$f$')
    # plt.hist(np.abs(error_0), 'gd', bins=50, label="$f - T_0 f$")
    # plt.hist(np.abs(error_1), 'rx', bins=50, label="$f - T_1 f$")
    # plt.hist(np.abs(error_2), 'ko', bins=50, label='$f - T_2 f$')
    # plt.hist(np.abs(f), bins=50, label='$f$')
    plt.hist(np.abs(error_0), bins=50, label="$|f - T_0 f|$")
    plt.hist(np.abs(error_1), bins=50, label="$|f - T_1 f|$")
    plt.hist(np.abs(error_2), bins=50, label='$|f - T_2 f|$')
    plt.legend(loc="best", fontsize=12)
    plt.xlabel("approximation errors", fontsize=16)
    plt.ylabel("frequencies", fontsize=16)
    filename = "figure/TaylorHist_" + str(iter) + ".pdf"
    fig.savefig(filename, format='pdf')
    plt.close()

    print("plot chance approximation")
    [xdata, chance, chance_0, chance_1, chance_2, l_beta, l_beta_0, l_beta_1, l_beta_2] = data["chance"]

    fig = plt.figure()
    plt.plot(xdata, chance, 'bs-', label='$I_{[0, \infty)}(f)$')
    plt.plot(xdata, chance_0, 'gd-', label="$I_{[0, \infty)}(T_0 f)$")
    plt.plot(xdata, chance_1, 'rx-', label="$I_{[0, \infty)}(T_1 f)$")
    plt.plot(xdata, chance_2, 'ko-', label='$I_{[0, \infty)}(T_2 f)$')
    # plt.legend()
    # plt.xlabel("# samples")
    # plt.ylabel("chance")
    # filename = "figure/Chance.pdf"
    # fig.savefig(filename, format='pdf')
    # plt.close()
    #
    # print("plot chance approximation")
    # fig = plt.figure()
    # xdata = np.power(2, range(N_s))
    plt.plot(xdata, l_beta, 'bs--', label=r'$\ell_{\beta}(f)$')
    plt.plot(xdata, l_beta_0, 'gd--', label=r"$\ell_{\beta}(T_0 f)$")
    plt.plot(xdata, l_beta_1, 'rx--', label=r"$\ell_{\beta}(T_1 f)$")
    plt.plot(xdata, l_beta_2, 'ko--', label=r'$\ell_{\beta}(T_2 f)$')
    plt.legend(loc="best", fontsize=12)
    plt.xlabel("# samples", fontsize=16)
    plt.ylabel("chance", fontsize=16)
    titlename = "step " + str(iter)
    plt.title(titlename, fontsize=16)
    plt.grid()
    filename = "figure/Chance_" + str(iter) + ".pdf"
    fig.savefig(filename, format='pdf')
    plt.close()

    [c_beta, l, l_beta, l_beta_0, l_beta_1, l_beta_2] = data["beta"]
    print("plot beta approximation")
    fig = plt.figure()
    # plt.plot(c_beta, l, 'b.--',label='func w/o')
    plt.plot(c_beta, l_beta, 'bs--', label=r'$\ell_{\beta}(f)$')
    plt.plot(c_beta, l_beta_0, 'gd-', label=r"$\ell_{\beta}(T_0 f)$")
    plt.plot(c_beta, l_beta_1, 'rx--', label=r"$\ell_{\beta}(T_1 f)$")
    plt.plot(c_beta, l_beta_2, 'ko--', label=r'$\ell_{\beta}(T_2 f)$')
    plt.legend(loc="best", fontsize=12)
    plt.xlabel(r'$\beta$', fontsize=16)
    # plt.ylabel(r'$\ell_{\beta}(f)$')
    titlename = "step " + str(iter)
    plt.title(titlename, fontsize=16)
    plt.grid()
    filename = "figure/Beta_" + str(iter) + ".pdf"
    fig.savefig(filename, format='pdf')
    plt.close()