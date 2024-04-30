import matplotlib.pyplot as plt
import numpy as np

output_name = "/home/wpan1/Documents/Data/tqg_example3/tqg_test"
energy = np.load(output_name + "energy.npy")
kinetic_energy = np.load(output_name + "kinetic_energy.npy")
potential_energy = np.load(output_name + "potential_energy.npy")

step = 120
plt.plot(energy[0:step], label='total energy', c='b')
plt.plot(kinetic_energy[0:step], label='kinetic energy', c='g')
plt.plot(potential_energy[0:step], label='potential energy', c='r')

plt.xlabel('time steps')
plt.ylabel('energy')
plt.grid(True)
plt.legend(loc='upper left')

plt.savefig("/home/wpan1/Documents/Data/tqg_example3/tqg_" + "all_energies_0_{}.png".format(step))

plt.show()
