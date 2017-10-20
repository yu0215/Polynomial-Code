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

s=90
r=100
t=120

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

#TODO: give Aenc[i],Benc[i] to worker i
#time the delivery (back and forth) and computing

#TODO: wait to mn workers and get:
#       1. list of workers completing the computation
#       2. results as a list of matrices
# example
lst=range(m*n)
Crtn=[Aenc[i]*(Benc[i].getT()) for i in range(m*n)]

#classical decoding algorithm

#for i in range(m*n):
    

end = datetime.now()
print (end - start)

