import matplotlib.pyplot as plt
import numpy as np

iterations = np.load("iterations.npy")

iter_avg = np.mean(iterations, axis=0)
print("iterations", iterations, "iter_avg", iter_avg)

plt.figure()

plt.plot(iterations[:,0], 'rx', label="reduced Newton for $u$")
plt.plot(iterations[:,1], 'bo', label="full Newton for $(u, \mu)$")
# plt.plot(iterations[:,2], 'kd', label="rescaled full Newton for $(u, \mu)$")
plt.xlabel("the N-th trial", fontsize=16)
plt.ylabel("# iterations", fontsize=16)
plt.legend(fontsize=16)
plt.savefig("full-reduced-comparison.pdf")
plt.close()