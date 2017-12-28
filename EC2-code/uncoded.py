#!/usr/bin/env python
'''
Uncoded matrix multiplication
'''

from mpi4py import MPI
import numpy as np
import random
import threading
import time

# Change to True for more accurate timing, sacrificing performance
barrier = True
# Change to True to imitate straggler effects
straggling = True

def loop():
  t = time.time()
  while time.time() < t + 60:
    a = 1 + 1

##################### Parameters ########################
# Use one master and N workers
N = 16

# Matrix division
m = 4
n = 4

# Field size assumed to be prime for this implementation
F = 65537

# Input matrix size - A: s by r, B: s by t
s = 4000
r = 4000
t = 4000
#########################################################

comm = MPI.COMM_WORLD

if comm.rank == 0:
  # Master
  print "Running with %d processes:" % comm.Get_size()

  # Decide and broadcast chosen straggler
  straggler = random.randint(1, N + 1)
  for i in range(N):
    comm.send(straggler, dest=i+1, tag=7)

  # Create random matrices of 8-bit ints
  A=np.matrix(np.random.random_integers(0,255,(r,s)))
  B=np.matrix(np.random.random_integers(0,255,(t,s)))

  # Split the matrices
  Ap=np.split(A,m)
  Bp=np.split(B,n)

  # Initialize return dictionary
  Crtn = []
  for i in range(N):
    Crtn.append(np.zeros((r/m, t/n), dtype=np.int_))

  # Start requests to send and receive
  reqA = [None] * N
  reqB = [None] * N
  reqC = [None] * N

  bp_start = time.time()

  for i in range(N):
    reqA[i] = comm.Isend([Ap[i%m], MPI.INT], dest=i+1, tag=15)
    reqB[i] = comm.Isend([Bp[i/m], MPI.INT], dest=i+1, tag=29)
    reqC[i] = comm.Irecv([Crtn[i], MPI.INT], source=i+1, tag=42)

  MPI.Request.Waitall(reqA)
  MPI.Request.Waitall(reqB)

  # Optionally wait for all workers to receive their submatrices, for more accurate timing
  if barrier:
    comm.Barrier()

  bp_sent = time.time()
  print "Time spent sending all messages is: %f" % (bp_sent - bp_start)
  
  MPI.Request.Waitall(reqC)
  bp_received = time.time()
  print "Time spent waiting for all workers is: %f" % (bp_received - bp_sent)

  # Verify correctness
  # Cver=[(Ap[i % m] * Bp[i / m].getT()) % F for i in range(m * n)]
  # print ([np.array_equal(Crtn[i], Cver[i]) for i in range(m * n)]) 

else:
  straggler = comm.recv(source=0, tag=7)

  Ai = np.empty_like(np.matrix([[0]*s for i in range(r/m)]))
  Bi = np.empty_like(np.matrix([[0]*s for i in range(t/n)]))
  rA = comm.Irecv(Ai, source=0, tag=15)
  rB = comm.Irecv(Bi, source=0, tag=29)

  rA.wait()
  rB.wait()

  if barrier:
    comm.Barrier()
  wbp_received = time.time()

  if straggling:
    if straggler == comm.rank:
      thread = threading.Thread(target=loop)
      thread.start()

  Ci = (Ai * (Bi.getT())) % F
  wbp_done = time.time()
  # print "Worker %d computing takes: %f\n" % (comm.Get_rank(), wbp_done - wbp_received)

  sC = comm.Isend(Ci, dest=0, tag=42)
  sC.Wait()
