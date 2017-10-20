#Computation at worker node:

import numpy as np
from datetime import datetime

#gets Ai, Bi, F from master
start = datetime.now()

Ci=(Ai*(Bi.getT()))%F
print (end - start)
#return Ci to master


