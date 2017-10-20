#test2: time the cost of computation stages with random matrices

import numpy as np
from datetime import datetime

s=10
r=10000
t=10000

start = datetime.now()

#creat random matrices of 8-bit ints
A=np.matrix(np.random.random_integers(0,255,(s,r)))
B=np.matrix(np.random.random_integers(0,255,(s,t)))


end = datetime.now()
print (end - start)


#compute AB^T
B=B.getT()

end = datetime.now()
print (end - start)

C=A*B

end = datetime.now()
print (end - start)
print C
