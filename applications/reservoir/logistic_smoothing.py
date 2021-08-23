import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2.,2.,1000)

kSet = [1, 2, 4, 8]
kLabel = [r'$\beta = 1$', r'$\beta = 2$', r'$\beta = 4$', r'$\beta = 8$']
kMarker = ['b-.', 'r--', 'g:', 'k-']
fig = plt.figure()
plt.plot(0, 0.5, 'mo')
plt.plot(x, (x>0), 'm.', linewidth=2, label='$I_{(-\infty, 0]}$')

for i in range(4):
    k = kSet[i]
    y = 1./(1+np.exp(-2*k*x))
    plt.plot(x,y, kMarker[i], label=kLabel[i])
plt.ylim([-0.1,1.1])
plt.legend(fontsize=16, loc='lower right')
plt.xlabel("$x$", fontsize=16)
plt.ylabel(r"$\ell_{\beta}(x)$", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

fig.savefig("figure/logistic_smooth.pdf")


plt.close()


fig = plt.figure()
kSet = [1, 2, 4, 8]
kLabel = ['$\gamma = 1$', '$\gamma = 2$', '$\gamma = 4$', '$\gamma = 8$']
for i in range(4):
    k = kSet[i]
    x = np.linspace(-2./np.sqrt(k), 2./np.sqrt(k), 1000)
    y = k/2.*((x>0)*x)**2
    plt.plot(x, y, kMarker[i], label=kLabel[i])

plt.ylim([-0.1, 2.1])
plt.legend(fontsize=16, loc='lower right')
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$\mathcal{S}_\gamma(x)$", fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

fig.savefig("figure/penalty_function.pdf")


plt.close()

