import generate as gn


alats40, energies40, _ = gn.vacancies1(nVacancies=40, xlim=[14.0, 20.0], nLines=401, n_times=50)
print("inished alats1")

from numpy import asarray
from numpy import savetxt

data_alats40 = asarray(alats40)
data_energies40 = asarray(energies40)

savetxt('data_alats40.csv', data_alats40, delimiter=',')
savetxt('data_energies40.csv', data_energies40, delimiter=',')
