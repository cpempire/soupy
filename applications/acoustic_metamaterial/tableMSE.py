
plotFlag = False

if plotFlag:
    import matplotlib.pyplot as plt

import numpy as np

type_set = ['random','constant','linear','quadratic','saa']

for type in type_set:

    data = np.load("data/mseResults"+type+".npz")
    Q_mean=data["Q_mean"]
    lin_diff_mean=data["lin_diff_mean"]
    quad_diff_mean=data["quad_diff_mean"]
    Q_var=data["Q_var"]
    lin_diff_var=data["lin_diff_var"]
    quad_diff_var=data["quad_diff_var"]

    N_test = len(Q_mean)

    step = N_test/10

    print("variance reduction at "+type+" design")
    header = ['# samples', '&', '$\hat{Q}$', '&',  'MSE($\hat{Q}$)', '&',  'MSE($Q-Q_{lin}$)', '&', 'MSE$(Q-Q_{quad}$)']
    print('{:<15} {:<5} {:<15} {:<5} {:<15} {:<5} {:<15} {:<5} {:<15}'.format(*header))
    for i in range(10):
        data = [(i+1)*step, '&', np.mean(Q_mean[0:(i+1)*step:1]),  '&',
                np.var(Q_mean[0:(i+1)*step:1])/((i+1)*step),  '&',
                np.var(lin_diff_mean[0:(i+1)*step:1])/((i+1)*step),  '&',
                np.var(quad_diff_mean[0:(i+1)*step:1])/((i+1)*step)]
        print('{:<15d} {:<5} {:<15.2e} {:<5} {:<15.2e} {:<5} {:<15.2e} {:<5} {:<15.2e}'.format(*data))

    header = ['# samples', '&', '$\hat{q}$', '&',  'MSE($\hat{q}$)', '&',  'MSE($q-q_{lin}$)', '&', 'MSE$(q-q_{quad}$)']
    print('{:<15} {:<5} {:<15} {:<5} {:<15} {:<5} {:<15} {:<5} {:<15}'.format(*header))
    for i in range(10):
        data = [(i+1)*step,'&', np.mean(Q_var[0:(i+1)*step:1]),'&',
                np.var(Q_var[0:(i+1)*step:1])/((i+1)*step),'&',
                np.var(lin_diff_var[0:(i+1)*step:1])/((i+1)*step),'&',
                np.var(quad_diff_var[0:(i+1)*step:1])/((i+1)*step)]
        print('{:<15d} {:<5} {:<15.2e} {:<5} {:<15.2e} {:<5} {:<15.2e} {:<5} {:<15.2e}'.format(*data))

    if plotFlag:
        plt.figure()
        lin_diff_mean_plot, = plt.semilogy(np.abs(lin_diff_mean/Q_mean), 'ko')
        quad_diff_mean_plot, = plt.semilogy(np.abs(quad_diff_mean/Q_mean), 'rx')
        plt.legend([lin_diff_mean_plot,quad_diff_mean_plot], ['linear mean error', 'quad mean error'])

        plt.figure()
        lin_diff_var_plot, = plt.semilogy(np.abs(lin_diff_var/Q_var), 'ko')
        quad_diff_var_plot, = plt.semilogy(np.abs(quad_diff_var/Q_var), 'rx')
        plt.legend([lin_diff_var_plot,quad_diff_var_plot], ['linear var error', 'quad var error'])

if plotFlag:
    plt.show()