import generate as gn


alats1, energies1, _ = gn.vacancies1(nVacancies=1, xlim=[14.0, 20.0], nLines=401, n_times=50)
print("inished alats1")

from numpy import asarray
from numpy import savetxt

data_alats1 = asarray(alats1)
data_energies1 = asarray(energies1)

savetxt('data_alats1.csv', data_alats1, delimiter=',')
savetxt('data_energies1.csv', data_energies1, delimiter=',')
