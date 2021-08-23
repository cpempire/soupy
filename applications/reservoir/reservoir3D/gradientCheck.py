import matplotlib.pyplot as plt
import numpy as np

k = 1.
c = 10.

# fig1 = plt.figure()
# fig2 = plt.figure()
fig1 = plt.figure(1)
fig2 = plt.figure(2)

for x0 in np.linspace(-1, 1, 10):

    l0 = 1./(1+np.exp(-2*k*x0))
    g0 = l0 - 0.5
    s0 = c/2.*((g0>0)*g0)**2

    dl = 2*k*np.exp(-2*k*x0)/(1+np.exp(-2*k*x0))**2
    ds = c*np.max([0, g0])*dl

    err = np.zeros(8)
    err_s = np.zeros(8)
    eps = np.power(0.5, range(8))

    for i in range(8):
        x1 = x0 + eps[i]

        l1 = 1./(1+np.exp(-2*k*x1))
        g1 = l1 - 0.5
        s1 = c / 2. * ((g1 > 0) * g1) ** 2

        err[i] = dl*eps[i] - (l1-l0)
        err_s[i] = ds*eps[i] - (s1-s0)

    plt.figure(1)
    plt.loglog(eps, np.abs(err),'.-')
    plt.figure(2)
    plt.loglog(eps, np.abs(err_s), '.-')

fig1.savefig("figure/gradientCheck_l.pdf")
fig2.savefig("figure/gradientCheck_s.pdf")

plt.close()

