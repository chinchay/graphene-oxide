import generate as gn


alats10, energies10, _ = gn.vacancies1(nVacancies=10, xlim=[14.0, 20.0], nLines=401, n_times=50)
print("inished alats1")

from numpy import asarray
from numpy import savetxt

data_alats10 = asarray(alats10)
data_energies10 = asarray(energies10)

savetxt('data_alats10.csv', data_alats10, delimiter=',')
savetxt('data_energies10.csv', data_energies10, delimiter=',')
