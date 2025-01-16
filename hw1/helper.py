import math
import numpy as np

k = 1.38 * 10**-23

def dB(n):
    return 10 * math.log10(n)

def un_dB(n):
    return 10 ** (n/10)

def two_ray(P, Ht, Hr, d, isdB): # P is dBm
    dBVal = P + dB((Ht*Ht*Hr*Hr)/(d*d*d*d)) # dBm
    dBVal = dBVal - 30 # Convert to dB
    if isdB:
        return dBVal
    else:
        return un_dB(dBVal)

def shadowing(P, Ht, Hr, d, isdB): # P is dBm
    # Use zero mean and std div = 6 log-normal distribution
    x = np.random.normal(loc=0.0, scale=6.0) # normal dist. if measured in dB

    print(f'x: {x}')
    dBVal = P + dB((Ht*Ht*Hr*Hr)/(d*d*d*d)) + x # dBm 

    dBVal = dBVal - 30 # Convert to dB
    if isdB:
        return dBVal
    else:
        return un_dB(dBVal)

def SNIR(S, T, B, I): 
    print(S, T, B, I)
    return dB(S/(k*T*B + I))
