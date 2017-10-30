#Computation at master node:

import numpy as np
from datetime import datetime
import Encoder 

start = datetime.now()

# use one master and N workers
N=4

#parameter
m=2
n=2

F=65537 #field size assumed to be prime for this implementation

#matrix input: A:s by r, B:s by t 

s=900
r=1000
t=1200

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


#TODO: give Ap[i],Bp[j] to worker (i+jm)
#time the delivery (back and forth) and computing

#TODO: wait for all
# example
Crtn=[(Ap[i%m]*Bp[i/m].getT())%F for i in range(m*n)]

end = datetime.now()
print (end - start)

#classical decoding algorithm

Jacob=[np.array([pow(lst[i],j,F) for j in range(m*n)]) for i in range(m*n)]

        
end = datetime.now()
print (end - start)

#verify correctness
Cver=[(Ap[i%m]*Bp[i/m].getT())%F for i in range(m*n)]
print ([np.array_equal(Crtn[i],Cver[i]) for i in range(m*n)])

