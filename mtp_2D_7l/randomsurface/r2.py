import generate as gn
import randomSurface as rs
from timeit import default_timer as timer


start = timer()

listA, listE = rs.getCurvesNotHighlyOrdrd(xlim=[14.0, 20.0], nLines=5, nRepeat=201)

end = timer()

#print(end - start)
f = open("timeittook2", "w")
f.write( "time = " + str( end - start ) )
f.close()


#listA, listE  = gn.plotCurvesRibbons(listnlines=[3, 5, 7, 15, 25, 31, 35, 41, 45, 51, 101, 401, 1001, 4001], xlim=[14.0, 20.0])
# https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# define data
data_listA = asarray(listA)
data_listE = asarray(listE)
# save to csv file
savetxt('data_listA2.csv', data_listA, delimiter=',')
savetxt('data_listE2.csv', data_listE, delimiter=',')


