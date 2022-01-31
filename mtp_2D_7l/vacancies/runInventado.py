import generate as gn


alats1, energies1, _ = gn.vacancies1(nVacancies=1, xlim=[14.0, 20.0], nLines=401, n_times=50)
print("finished alats1")

_,      energies2, _ = gn.vacancies1(nVacancies=2, xlim=[14.0, 20.0], nLines=401, n_times=50)
print("finished alats2")

_,      energies3, _ = gn.vacancies1(nVacancies=3, xlim=[14.0, 20.0], nLines=401, n_times=50)
print("finished alats3")

_,      energies4, _ = gn.vacancies1(nVacancies=4, xlim=[14.0, 20.0], nLines=401, n_times=50)
print("finished alats4")

_,      energies5, _ = gn.vacancies1(nVacancies=5, xlim=[14.0, 20.0], nLines=401, n_times=50)
print("finished alats5")

from numpy import asarray
from numpy import savetxt

data_alats1 = asarray(alats1)
data_energies1 = asarray(energies1)
data_energies2 = asarray(energies2)
data_energies3 = asarray(energies3)
data_energies4 = asarray(energies4)
data_energies5 = asarray(energies5)

savetxt('data_alats1.csv', data_alats1, delimiter=',')
savetxt('data_energies1.csv', data_energies1, delimiter=',')
savetxt('data_energies2.csv', data_energies2, delimiter=',')
savetxt('data_energies3.csv', data_energies3, delimiter=',')
savetxt('data_energies4.csv', data_energies4, delimiter=',')
savetxt('data_energies5.csv', data_energies5, delimiter=',')



