#Computation at master node:


#fast decoding algorithm: hardcoded for m=n=4, F=65537=2**(2**sig)+1
#changes: encoding, direct delivery to final destination,verification



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


#s=4
#r=4
#t=4


#creat random matrices of 8-bit ints
A=np.matrix(np.random.random_integers(0,255,(r,s)))
B=np.matrix(np.random.random_integers(0,255,(t,s)))

#test
#A=np.matrix([[140, 196, 202,  90], [ 54, 255,  71, 206], [195, 186, 187, 155], [ 71, 111, 127,  25]])
#B=np.matrix([[215, 215, 110,  23], [124,  83,   0,  58], [ 31,   3, 220,  83], [181,  86, 103,  11]])

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
var=[pow(64,i,65537) for i in range(16)]+[3,9]#at most 18 distinct roots

Aenc=[sum([Ap[j]*(pow(var[i],j,F)) for j in range(m)])%F for i in range(N)] 
Benc=[sum([Bp[j]*(pow(var[i],(j*m),F)) for j in range(n)])%F for i in range(N)] 

#print Aenc[17]
#print Benc[17]
#print (Aenc[17]*(Benc[17].getT()))%F


end = datetime.now()
print (end - start)
#TODO: give Aenc[i],Benc[i] to worker i
#time the delivery (back and forth) and computing

#TODO: wait to mn workers and get:
#       1. list of workers completing the computation
#       2. results as a list of matrices
# example
lst=random.sample(range(N), m*n)

#testcase
lst=[7, 4, 5, 13, 6, 15, 9, 14, 16, 10, 12, 17, 8, 2, 1, 3]

Crtn=[[] for i in range(N)]

for i in lst:
    Crtn[i]=(Aenc[i]*(Benc[i].getT()))%F

#print Crtn 

end = datetime.now()
print (end - start)

#fast decoding algorithm: hardcoded for m=n=4, F=65537=2**(2**sig)+1
sig=4
xlist=[var[i] for i in lst]


#if (9 in xlist) or (3 in xlist): #recover using 3 or 9
missing=set(range(m*n))-set(lst)

#print missing

#debug
#coeff=[1]*(m*n)
#for j in range(m*n):#compute coefficient
#    for k in set(lst)-set([lst[j]]):
#        coeff[j]=(coeff[j]*(var[0]-var[k])*pow(var[lst[j]]-var[k],F-2,F))%F
#print [Crtn[lst[j]] for j in range(16)]
#print coeff
#print sum([(Crtn[lst[j]]*coeff[j])%F for j in range(16)])%F

#debug end


for i in missing:
    coeff=[1]*(m*n)
    for j in range(m*n):#compute coefficient
        for k in set(lst)-set([lst[j]]):
            coeff[j]=(coeff[j]*(var[i]-var[k])*pow(var[lst[j]]-var[k],F-2,F))%F
    #print coeff
    Crtn[i]=sum([Crtn[lst[j]]*coeff[j] for j in range(16)])%F


end = datetime.now()
print "Find Missing Done"
print (end - start)    
#print Crtn    
#FFT
#2^-1=32769

def NFT(A,l,F,om):#input,length,field size, primitive root, assume len=2^int
    if l==1:
        return 0
    for i in range(l/2):
        A[i]=((A[i]+A[i+l/2])*(32769))%F
        A[i+l/2]=((A[i]-A[i+l/2])*pow(om,F-1-i,F))%F
        NFT([A[j] for j in range(l/2)],l/2,F,om)
        NFT([A[j+l/2] for j in range(l/2)],l/2,F,om)
    return 0


#inplace FFT
#def NFT(A,l,F,om):#input,length,field size, primitive root, assume len=2^int
#    if l==1:
#        return 0
#    for i in range(l/2):
#        A[i]=((A[i]+A[i+l/2])*(32769))%F
#        A[i+l/2]=((A[i]-A[i+l/2])*pow(om,F-1-i,F))%F
#        NFT([A[j] for j in range(l/2)],l/2,F,om)
#        NFT([A[j+l/2] for j in range(l/2)],l/2,F,om)
#    return 0


#NFT([Crtn[i] for i in range(m*n)],m*n,F,rt)


end = datetime.now()
print (end - start)

#verify correctness

Cver=[(Ap[i%n]*Bp[i/n].getT())%F for i in range(m*n)]
print ([np.array_equal(Crtn[i],Cver[i]) for i in range(m*n)])

#print Cver
#print Crtn
#print missing
