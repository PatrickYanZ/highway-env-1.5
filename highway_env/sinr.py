
import math
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from scipy import linalg
import numpy.matlib
import random

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
# print(arr_num)

Rate = 5e9 
Wt=5e8
Wr=40e6
NP = 10e-10
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
    # arr = []
    # for i in range(len(matrix)):
    #     arr.append(np.sum(matrix[i]))
    # # arr = arr.transpose()
    # return arr
    return matrix.sum(axis=0)


def rf_sinr_matrix(distance_matrix,vehicles,bss):
    """
    Convert distance matrix to sinr matrix

    """
    NU,NRF = distance_matrix.shape # row is vehicle, column is rf bs
    # print(distance_matrix)
    # print("distance_matrix.shape",NU,NRF )
    # NU = len(distance_matrix) 
    # NRF = len(distance_matrix[0])

    d_matrix = np.array(distance_matrix)

    fadeRand = generate_exponential_matrix(1,NRF,NU)
    # print("fade rand shape",np.shape(fadeRand))

    #signal matrix for RF
    # SRF = gammaI*fadeRand*PR*d_matrix
    # SRF = np.dot(gammaI,fadeRand)
    SRF = np.multiply(gammaI,fadeRand)
    # print("srf 1",np.shape(SRF))

    # SRF = np.dot(SRF,PR)
    SRF = np.multiply(SRF,PR)
    # print("srf 2",np.shape(SRF))

    # SRF = np.dot(SRF,d_matrix)
    SRF = np.multiply(SRF,np.transpose(d_matrix))
    # print("srf 3",np.shape(SRF))
    '''
     SRF = np.linalg.matrix_power(SRF,-1*alpha)
    '''
    # SRF = fractional_matrix_power(SRF,-1*alpha)
    SRF = SRF **(-1*alpha) #fractional_matrix_power(distance_matrix_thz,2)
    # print("srf 4",np.shape(SRF))
    
    # interference : interf=repmat(sum(SRF,1),NRF,1)-SRF; %interference for RF
    sum_srf = sum_of_each_row(SRF)
    # print("sum_srf",np.shape(sum_srf))

    interf=np.matlib.repmat(sum_srf,NRF,1)
    # print("interf",np.shape(interf))

    interf=np.subtract(interf, SRF)
    
    # print("interf shape",np.shape(interf))
    # print("interf",interf)

    #power from all base-stations to all users
    NP=10e-10 #(10) ** (-10)
    RPrAllu1 = Wr * np.log2(np.add(1,np.divide(SRF,np.add(NP, interf))))
    # print(RPrAllu1)
    RPrAllu1 = np.transpose(RPrAllu1)
    interf=np.transpose(interf)
    # print(RPrAllu1.shape)


    ## column row names should be recovered ### 
    # print(distance_matrix) 
    # print(d_matrix)
    # print(RPrAllu1)
    # print('vehicle list is ', vehicles,)
    # print('bs_list is',bss)

    sinr_matrix = pd.DataFrame(RPrAllu1 , columns = bss, index = vehicles)
    interf_matrix = pd.DataFrame(interf , columns = bss, index = vehicles)
    # print(df)

    return sinr_matrix,interf_matrix


def thz_sinr_matrix(distance_matrix,vehicles,bss):
    """
    Convert distance matrix to sinr matrix

    """
    
    NU,NRF = distance_matrix.shape # row is vehicle, column is rf bs
    # print(distance_matrix)
    # print("distance_matrix.shape",NU,NRF )
    # NU = len(distance_matrix) 
    # NRF = len(distance_matrix[0])

    d_matrix = np.array(distance_matrix)

    fadeRand = generate_exponential_matrix(1,NRF,NU)
    # print("fade rand shape",np.shape(fadeRand))

    #signal matrix for RF
    # SRF = gammaI*fadeRand*PR*d_matrix
    # SRF = np.dot(gammaI,fadeRand)
    SRF = np.multiply(gammaI,fadeRand)
    # print("srf 1",np.shape(SRF))

    # SRF = np.dot(SRF,PR)
    SRF = np.multiply(SRF,PR)
    # print("srf 2",np.shape(SRF))

    # SRF = np.dot(SRF,d_matrix)
    SRF = np.multiply(SRF,np.transpose(d_matrix))
    # print("srf 3",np.shape(SRF))
    '''
     SRF = np.linalg.matrix_power(SRF,-1*alpha)
    '''
    # SRF = fractional_matrix_power(SRF,-1*alpha)
    SRF = SRF **(-1*alpha) #fractional_matrix_power(distance_matrix_thz,2)
    # print("srf 4",np.shape(SRF))
    
    # interference : interf=repmat(sum(SRF,1),NRF,1)-SRF; %interference for RF
    sum_srf = sum_of_each_row(SRF)
    # print("sum_srf",np.shape(sum_srf))

    interf=np.matlib.repmat(sum_srf,NRF,1)
    # print("interf",np.shape(interf))

    interf=np.subtract(interf, SRF)
    # print("interf shape",np.shape(interf))
    # print("interf",interf)

    #power from all base-stations to all users
    NP=10e-10 #(10) ** (-10)
    RPrAllu1 = Wr * np.log2(np.add(1,np.divide(SRF,np.add(NP, interf))))
    # print(RPrAllu1)
    # print(RPrAllu1.shape)
    RPrAllu1 = np.transpose(RPrAllu1)
    interf=np.transpose(interf)
    # print(RPrAllu1.shape)


    ## column row names should be recovered ### 
    # print(distance_matrix) 
    # print(d_matrix)
    # print(RPrAllu1)
    # print('vehicle list is ', vehicles,)
    # print('bs_list is',bss)

    sinr_matrix = pd.DataFrame(RPrAllu1 , columns = bss, index = vehicles)
    interf_matrix = pd.DataFrame(interf , columns = bss, index = vehicles)
    # print(df)

    return sinr_matrix,interf_matrix

def sinr_with_threshold(sinr_matrix,bs_assignment):
    '''
    Input 
    1.sinr matrix (merged rf thz version)
    2.bs assignment matrix

    we devide the sinr matrix to the bss
 	                rf1	rf2	rf3	rf4	rf5	rf6	rf7	th1	th2	th3	th4	th5	th6
sinr	                10	20	30	20	20	40	10	20	30	40	40	30	20
bss assignment	    4	8	9	9	2	10	3	3	3	2	4	1	3
data with threshold	    2.5	2.5	3.333333	2.222222	10	4	3.333333	6.666667	10	20	10	30	6.666667



    Output
    sinr matrix with threshold


    '''
    sinr_matrix_with_threshold = sinr_matrix.div(bs_assignment.sum(), axis=0)#,index=sinr_matrix.columns

    return sinr_matrix_with_threshold

    

# def getInterf(sinr_matrix):
    
#     '''
#     #axis=1 specifies that the sum will be done on the rows.
#     # Suppose the vehicle can connect with one bs with maximum SINR
#     # The interference is the sum of all SIMR for one vihele deduct the maximum SINR
#     '''
#     intf = sinr_matrix
#     intf["Interference"] = intf.sum(axis=1) - intf.max(axis=1)
#     intf1 = intf[['Interference']]
#     return intf1

# def get_signal_rf(distance_matrix,vehicles,bss):
#     '''
#     # seperate these two base station list 
#     # substitude the dimension of NRF(number of Rf base station) number of vihicle seperate
#     # For Data Hertz
#     # No fading in THz BSs
#     '''


#     #fadeRand = exprnd(1,NRF,UoI);
#     # cars_db_path = "E:/research/trafficModel-master/2021_05_27_23_27_output.csv"
#     # car_db_obj = pd.read_csv(cars_db_path) #read car database
#     # df_rsdb = pd.read_csv("RFBSDB.csv")

#     NOV,NRF = distance_matrix.shape
#     # NRF = df_rsdb['bs_id'].size
#     # print("NRF is",NRF)
#     # NOV = df_cars_row_count
#     # print("NOV is",NOV)

#     # print("faderand")
#     # fadeRand = random.exponential(scale=1, size=(NOV,NRF)) # this number of Rf base station, number of vehicles.
#     fadeRand = generate_exponential_matrix(1,NOV,NRF)
#     # print(fadeRand)
#     # print(len(fadeRand), len(fadeRand[0]))

#     #fadeRand = random.exponential(scale=1)
#     # print("SRF here")
#     SRF = np.multiply(gammaI,fadeRand)
#     # print(len(SRF), len(SRF[0]))
#     #SRF = np.multiply(SRF,PR)  

#     #SRF = gammaI * fadeRand *PR
#     # print("srf",SRF)
#     #matrix = getDIstanceMatrix("E:/research/trafficModel-master/result/2021_06_01_14_35_11.csv","E:/research/trafficModel-master/BSDB.csv")
#     # matrix = getDIstanceMatrix(cars_db_path,"E:/research/trafficModel-master/RFBSDB.csv")
#     matrix = np.array(distance_matrix)
#     # matrix = np.array(distance_matrix)

#     # print("distance matrix")
#     # print(matrix)

#     # matrix = matrix.power(-alpha)
#     matrix = np.power(matrix,-alpha)
#    # print(len(matrix), len(matrix[0]))
#     #2021_05_27_23_27_output E:\research\trafficModel-master\2021_05_27_23_27_output.csv
#     #SRF = np.multiply(SRF,D_ue_Rbs) 
#     #matrix.mul(SRF)
#     matrix = np.multiply(matrix,SRF)#matrix * SRF  signal/power

#     # return matrix
#     # print('get distance matrix shape ',distance_matrix.shape,'get rf matrix shape ',matrix.shape)
#     df = pd.DataFrame(matrix , columns = bss, index = vehicles)
#     return df

# def get_signal_thz(distance_matrix_thz,vehicles,bss_thz):
#     '''
#     # seperate these two base station list 
#     # substitude the dimension of NRF(number of Rf base station) number of vihicle seperate
#     # For Data Hertz
#     # No fading in THz BSs
#     '''

#     NOV,NThz = distance_matrix_thz.shape

#     fadeRand1 = generate_exponential_matrix(1,NOV,NThz)

#     STHz = np.multiply(gammaII,fadeRand1)
#     STHz = np.multiply(STHz,PT)

#     matrix = np.array(distance_matrix_thz)
#     matrix = np.power(matrix,-kf)
#     # matrix = linalg.expm(matrix) # exponential matrix https://www.tutorialspoint.com/python-scipy-linalg-expm
#     matrix = np.exp(matrix)

#     STHz = np.multiply(STHz,matrix)

#     temp_dis_matrix = distance_matrix_thz **2 #fractional_matrix_power(distance_matrix_thz,2)
#     STHz = np.divide(STHz,temp_dis_matrix)
#     matrix = STHz
#     df = pd.DataFrame(matrix , columns = bss_thz, index = vehicles)
#     return df

# def getSINR(signal_matrix,Interf):
#     # rows_num =  len(signal_matrix)
#     cols_num = len(signal_matrix.columns)

#     for i in range(cols_num-1) :
#         tem_col_name = 'col_' +str(i)
#         Interf[tem_col_name]=Interf['Interference']
#         # Interf.loc[tem_col_name]=Interf['Interference']

#     sinr_rf = np.divide(signal_matrix,Interf) 
#     sinr_rf = sinr_rf.drop(['Interference'], axis=1)
  
#     # print(signal_matrix.shape)
#     return(sinr_rf)

# def get_rf_sinr(distance_matrix,vehicles,bss):
#     signal_matrix_rf = get_signal_rf(distance_matrix,vehicles,bss)
#     interf_rf = getInterf(signal_matrix_rf)
#     sinr_rf = getSINR(signal_matrix_rf,interf_rf)
#     # df = pd.DataFrame(sinr_rf , columns = bss, index = vehicles)
   
#     return sinr_rf,interf_rf


# def get_thz_sinr(distance_matrix,vehicles,bss):
#     signal_matrix_thz = get_signal_thz(distance_matrix,vehicles,bss)
#     interf_thz = getInterf(signal_matrix_thz)
#     sinr_thz = getSINR(signal_matrix_thz,interf_thz)
#     # df = pd.DataFrame(sinr_rf , columns = bss, index = vehicles)
#     return sinr_thz,interf_thz

# def get_rf_dr(distance_matrix,vehicles,bss):
#     sinr_rf,interf_rf = get_rf_sinr(distance_matrix,vehicles,bss)
#     # RPrAllu1 = Wr * np.log2(np.add(1,np.divide(sinr_rf,np.add(NP, interf_rf))))
#     print("sinr rf shape",type(sinr_rf) ,(sinr_rf.shape))
#     print("intef rf shape",type(interf_rf) ,(interf_rf.shape))
#     print(sinr_rf)
#     print(interf_rf)
#     # print(interf_rf.columns)
#     interf_rf = interf_rf.drop(['Interference'], axis=1) # keep (21, 19)
#     # interf_rf.drop(columns='Interference') # keep (21, 19)
#     print("after")
#     print(interf_rf)
#     RPrAllu1 = np.multiply(Wr,np.log2(np.add(1, (np.divide(sinr_rf, (NP+interf_rf)))))) 
#     df = pd.DataFrame(RPrAllu1 , columns = bss, index = vehicles)
#     # RPrAllu1 = Wr * log2(1+SRF./(NP+interf));
#     return df

# def get_thz_dr(distance_matrix,vehicles,bss):
#     sinr_thz,interf_thz = get_thz_sinr(distance_matrix,vehicles,bss)
#     # RPrAllu1 = Wr * np.log2(np.add(1,np.divide(sinr_rf,np.add(NP, interf_rf))))
#     TPrAllu1 = Wt * np.log2(1 + (sinr_thz / (NP+interf_thz)))
#     TPrAllu1 = Wt * np.log2(1 + (sinr_thz / (NP + interf_thz)))



    # df = pd.DataFrame(TPrAllu1 , columns = bss, index = vehicles)
    # # RPrAllu1 = Wr * log2(1+SRF./(NP+interf));
    # return df
