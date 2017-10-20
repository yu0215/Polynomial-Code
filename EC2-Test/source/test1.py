#test1: read data from a file and compute the matrix product

import numpy as np
from datetime import datetime

start = datetime.now()

F=open("data","r") #read data

s=int(F.readline())
r=int(F.readline())
t=int(F.readline())

#A=np.zeros(shape=(s,r))
#B=np.zeros(shape=(s,t))

#read A
A=[]
for i in range(s):
    A.append([int(x) for x in F.readline().strip().split(' ')])
A=np.matrix(A)

B=[]
for i in range(s):
    B.append([int(x) for x in F.readline().strip().split(' ')])
B=np.matrix(B)

#compute A^TB
C=A.getT()*B

end = datetime.now()
print (end - start)
