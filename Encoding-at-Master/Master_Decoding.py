#Computation at master node:


#fast decoding algorithm: hardcoded for m=n=4, F=65537=2**(2**sig)+1

import numpy as np
from datetime import datetime
import Encoder 
import random

start = datetime.now()

# use one master and N workers
N=18

#parameter
m=4
n=4

F=65537 #field size assumed to be prime for this implementation

#matrix input: A:s by r, B:s by t 

s=4000
r=4000
t=4000

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


#fast decoding algorithm: hardcoded for m=n=4, F=65537=2**(2**sig)+1
#8th primitive root 64
rt=64
var=[pow(64,i,65537) for i in range(16)]+[3,9]#18 distinct roots

Aenc=[sum([Ap[j]*(i**(j*1)) for j in range(m)])%F for i in var] 
Benc=[sum([Bp[j]*(i**(j*m)) for j in range(n)])%F for i in var] 


end = datetime.now()
print (end - start)
#TODO: give Aenc[i],Benc[i] to worker i
#time the delivery (back and forth) and computing

#TODO: wait to mn workers and get:
#       1. list of workers completing the computation
#       2. results as a list of matrices
# example
lst=random.sample(range(N), m*n)


Crtn=[(Aenc[i]*(Benc[i].getT()))%F for i in lst]

end = datetime.now()
print (end - start)

#fast decoding algorithm: hardcoded for m=n=4, F=65537=2**(2**sig)+1
sig=4

Jacob=[np.array([pow(var[lst[i]],j,F) for j in range(m*n)]) for i in range(m*n)]


#Forward Gaussian Elimination
for i in range(m*n):
    for j in range(i+1,m*n):
        Crtn[j]=((Crtn[j]-Crtn[i])*pow(var[lst[j]]-var[lst[i]],F-2,F))%F
        Jacob[j]=((Jacob[j]-Jacob[i])*pow(var[lst[j]]-var[lst[i]],F-2,F))%F
#Backward Gaussian Elimination
for i in range(m*n-1,0,-1):
    for j in range(i):
        Crtn[j]=(Crtn[j]-Crtn[i]*Jacob[j][i])%F
        
end = datetime.now()
print (end - start)

#verify correctness

Cver=[(Ap[i%n]*Bp[i/n].getT())%F for i in range(m*n)]
print ([np.array_equal(Crtn[i],Cver[i]) for i in range(m*n)])
