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

    dBVal = P + dB((Ht*Ht*Hr*Hr)/(d*d*d*d)) + x # dBm 

    dBVal = dBVal - 30 # Convert to dB
    if isdB:
        return dBVal
    else:
        return un_dB(dBVal)

def SNIR(S, T, B, I): 
    # S, I are in W
    return dB(S/(k*T*B + I))

def gen_hexagon(n, r, x, y):
    # Uniformly plot n points in hexagon with radius r centered at (x, y) using rejection sampling
    xlist = []
    ylist = []
    i = 0
    half_L = math.sqrt(3) * r / 2
    while i < n:
        x0 = np.random.uniform(-r, r)
        y0 = np.random.uniform(-half_L, half_L)
        if y0 < -math.sqrt(3) * (x0-r) and y0 > math.sqrt(3) * (x0-r) and y0 < math.sqrt(3) * (x0+r) and y0 > -math.sqrt(3) * (x0+r):
            xlist.append(x0 + x)
            ylist.append(y0 + y)
            i += 1
    return xlist, ylist

def hexagon_vertices(r, x, y):
    angles = np.linspace(0, 2*np.pi, 7)
    xlist = r * np.cos(angles) 
    xlist = [x + i for i in xlist]
    ylist = r * np.sin(angles)
    ylist = [y + i for i in ylist]
    return xlist, ylist

def grid_generator(n, r, coords, vec_i, vec_j):
    # Generate a grid of hexagons with radius r
    hex_xlist = []
    hex_ylist = []
    hex_device_x = []
    hex_device_y = []
    base_x = []
    base_y = []
    for i, j in coords:
        cen = i*vec_i + j*vec_j
        x0, y0 = hexagon_vertices(r, cen[0], cen[1])
        device_x, device_y = gen_hexagon(n, r, cen[0], cen[1])
        hex_xlist.append(x0)
        hex_ylist.append(y0)
        hex_device_x.append(device_x)
        hex_device_y.append(device_y)
        base_x.append(cen[0])
        base_y.append(cen[1])
    return hex_xlist, hex_ylist, hex_device_x, hex_device_y, base_x, base_y

def flatten(l):
    return [item for sublist in l for item in sublist]
