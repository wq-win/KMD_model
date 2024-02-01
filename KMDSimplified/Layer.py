from Synapse import Synapse
from Model import KMDmodel


kmd = KMDmodel()
n_K=6
n_M=3
n_D=3
KCtoMBON = Synapse(n_K, n_M)
KCtoDAN = Synapse(n_K, n_D)
DANtoKC = Synapse(n_D, n_K)