#Computation at master node:

import numpy as np
from datetime import datetime
import Encoder 

start = datetime.now()

# use one master and N workers
N=5

#parameter
m=2
n=2

F=65537 #field size assumed to be prime for this implementation

#matrix input: A:s by r, B:s by t 

s=9
r=10
t=8

#creat random matrices of 8-bit ints
A=np.matrix(np.random.random_integers(0,255,(r,s)))
B=np.matrix(np.random.random_integers(0,255,(t,s)))

#polynomial code
#split the matrices

Ap=np.split(A,m)
Bp=np.split(B,n)

#encode the matrices

end = datetime.now()
print (end - start)

Aenc=Encoder.Enc(Ap,N,m,1,F)
Benc=Encoder.Enc(Bp,N,n,m,F)


end = datetime.now()
print (end - start)
#TODO: give Aenc[i],Benc[i] to worker i
#time the delivery (back and forth) and computing

#TODO: wait to mn workers and get:
#       1. list of workers completing the computation
#       2. results as a list of matrices
# example
lst=range(m*n)
Crtn=[Aenc[i]*(Benc[i].getT()) for i in range(m*n)]

end = datetime.now()
print (end - start)

#classical decoding algorithm

Jacob=[np.array([pow(lst[i],j,F) for j in range(m*n)]) for i in range(m*n)]

#Forward Gaussian Elimination
for i in range(m*n):
    for j in range(i+1,m*n):
        Crtn[j]=((Crtn[j]-Crtn[i])*pow(lst[j]-lst[i],F-2,F))%F
        Jacob[j]=((Jacob[j]-Jacob[i])*pow(lst[j]-lst[i],F-2,F))%F
#Backward Gaussian Elimination
for i in range(m*n-1,0,-1):
    for j in range(i):
        Crtn[j]=(Crtn[j]-Crtn[i]*Jacob[j][i])%F
        
end = datetime.now()
print (end - start)

#verify correctness

Cver=[(Ap[i%n]*Bp[i/n].getT())%F for i in range(m*n)]
print ([np.array_equal(Crtn[i],Cver[i]) for i in range(m*n)])
