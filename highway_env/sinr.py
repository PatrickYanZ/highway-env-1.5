
import math
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
import numpy.matlib

pi = math.pi

## =================== RF Parameters ======================================
alpha =2.5                   #%Path Loss Exponent better to have indoor small
PR = 1
fcRF=2.1e9
GT=1
GR=1
gammaI=(3e8)**2 * GT * GR/(16 * pi ** 2 * fcRF **2);  # %used in sinr 


## ===================== THz Parameters ===========================
PT = 1                          #   % Tranmitted Power
kf = 0.05                        #   % Absorbtion loss
fcTH=1.0e12
GTT=316.2
GRR=316.2
# GTT=31.62;
# GRR=31.62

thetabs=pi/6;#%%%in degrees
thetamt=pi/6;#%%%in degrees
FBS=thetabs/(2*pi)#;
FMT=thetamt/(2*pi)#;
prob= FBS*FMT#;
gammaII=(3e8)**2 * GTT * GRR /(16 * pi ** 2 * fcTH ** 2)#;
 

R_max = 100#;
Nit = 10000#;
lambdas=10#;

## ===================== Rate and SINR theshold calculation ========================================
arr_num = [0] * 5
print(arr_num)

Rate = 5e9 
Wt=5e8
Wr=40e6
#SINRthRF =2.^(Rate./(Wr))-1 #%thresholds
#SINRthTH =2.^(Rate./(Wt))-1

#Bias=[1000 100 1 0.001 0.0001 0.0001 0.00001]#%%%0.05
#%Bias=[ 10^6  10^5 10^4  10^4 10^3 10^3 10^3];%%%0.2

def generate_exponential_matrix(miu,m,n):
    """
    Generate Exponential matrix with mean miu with m rows and n columns
    ref https://www.geeksforgeeks.org/numpy-random-exponential-in-python/
    """
    exponential_matrix = []
    for i in range(m):
        row = np.random.exponential(miu, n)
        exponential_matrix.append(row)
    return exponential_matrix

def sum_of_each_row(matrix):
    arr = []
    for i in range(len(matrix)):
        arr.append(np.sum(matrix[i]))
    # arr = arr.transpose()
    return arr


def rf_sinr_matrix(distance_matrix,vehicles,bss):
    """
    Convert distance matrix to sinr matrix

    """
    NU,NRF = distance_matrix.shape # row is vehicle, column is rf bs
    print("distance_matrix.shape",NU,NRF )
    # NU = len(distance_matrix) 
    # NRF = len(distance_matrix[0])

    d_matrix = np.array(distance_matrix)

    fadeRand = generate_exponential_matrix(1,NRF,NU)

    #signal matrix for RF
    SRF = np.dot(gammaI,fadeRand)
    SRF = np.dot(SRF,PR)
    SRF = np.dot(SRF,d_matrix)
    '''
     SRF = np.linalg.matrix_power(SRF,-1*alpha)
    '''
    SRF = fractional_matrix_power(SRF,-1*alpha)
    
    # interference : interf=repmat(sum(SRF,1),NRF,1)-SRF; %interference for RF
    sum_srf = sum_of_each_row(SRF)
    interf=np.matlib.repmat(sum_srf,NRF,1)
    interf=np.subtract(interf, SRF)

    #power from all base-stations to all users
    NP=10e-10 #(10) ** (-10)
    RPrAllu1 = Wr * np.log2(np.add(1,np.divide(SRF,np.add(NP, interf))))


    ## column row names should be recovered ### 
    # print(distance_matrix) 
    # print(d_matrix)
    print(RPrAllu1)
    print('vehicle list is ', vehicles,)
    print('bs_list is',bss)

    df = pd.DataFrame(RPrAllu1 , columns = vehicles, index = bss)
    print(df)

    return df
