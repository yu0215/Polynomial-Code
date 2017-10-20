#Encoder at master mode:

import numpy as np

#input Mp,N
def Enc(Mp,N,div,k,F):
    return [sum([Mp[j]*(i**(j*k)) for j in range(div)])%F for i in range(N)]

#Note use a finite field of size 65537
