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

def urban_generator(n, r, coords, vec_i, vec_j):
    # First find the center of Central BS
    isd = math.sqrt(3) * r
    # (0, isd) is starting direction, first walk to that direction x 3 and turn counter clockwise pi/3, walk another 2
    center = np.array([0, 0])
    cur = np.array([0, isd])
    all_grids = []
    hex_xlist, hex_ylist, _, _, base_x, base_y = grid_generator(n, r, coords, vec_i, vec_j)
    all_grids.append([hex_xlist, hex_ylist, base_x, base_y]) # Central of central BS
    for i in range(6):
        # Rotate direction counter clockwise by pi/3
        rotated = np.array([cur[0]*math.cos(math.pi/3) - cur[1]*math.sin(math.pi/3), cur[0]*math.sin(math.pi/3) + cur[1]*math.cos(math.pi/3)])
        center = cur * 3 + rotated * 2
        hex_xlist, hex_ylist, _, _, base_x, base_y = grid_generator(n, r, coords, vec_i, vec_j)
        hex_xlist = [[i + center[0] for i in j] for j in hex_xlist]
        hex_ylist = [[i + center[1] for i in j] for j in hex_ylist]
        base_x = [i + center[0] for i in base_x]
        base_y = [i + center[1] for i in base_y]
        all_grids.append([hex_xlist, hex_ylist, base_x, base_y])
        cur = rotated
    return all_grids
    # Generate 6 grids and add center as offset

def random_walk(minSpeed, maxSpeed, minT, maxT, simulation_time, initx, inity):
    # random direction in [0, 2pi]
    # random speed in [minSpeed, maxSpeed]
    # random time in [minT, maxT], time should be integer
    elapsed = 0
    walk_locations = [[initx, inity, 0]]
    while elapsed < simulation_time:
        direction = np.random.uniform(0, 2*np.pi)
        speed = np.random.uniform(minSpeed, maxSpeed)
        time = np.random.randint(minT, maxT)
        # print(f'Direction: {direction}, Speed: {speed}, Time: {time}')
        for i in range(time):
            newx = walk_locations[-1][0] + speed * math.cos(direction)
            newy = walk_locations[-1][1] + speed * math.sin(direction)
            walk_locations.append([newx, newy, i + elapsed])
        elapsed += time
       
    return walk_locations

def dl_evaluate_handoff(all_base_coord, ms_x, ms_y, two_ray_lambda, SINR_lambda):
    powers = []
    for i in range(len(all_base_coord)):
        all_base_x, all_base_y, base_i = all_base_coord[i]
        d = math.sqrt((all_base_x - ms_x) ** 2 + (all_base_y - ms_y) ** 2)
        powers.append(two_ray_lambda(d))
    total_intf = sum(powers)

    sinr_best = -2**31
    sinr_best_bs = -1
    for i in range(len(powers)):
        interference_at_bs = total_intf - powers[i]
        sinr = SINR_lambda(powers[i], interference_at_bs)
        if sinr > sinr_best:
            sinr_best = sinr
            sinr_best_bs = all_base_coord[i][2]
    
    # print(f'Best SINR: {sinr_best}, Best BS: {sinr_best_bs+1}')
    return sinr_best_bs


def ul_evaluate_single_bs(all_ms_coord, bs_x, bs_y, two_ray_lambda, SINR_lambda):
    # returns SINR for all MS based on single BS
    powers = []
    for i in range(len(all_ms_coord)):
        ms_x, ms_y = all_ms_coord[i]
        d = math.sqrt((bs_x - ms_x) ** 2 + (bs_y - ms_y) ** 2)
        powers.append(two_ray_lambda(d))
    total_intf = sum(powers)

    all_sinr = []
    for i in range(len(powers)):
        interference_at_bs = total_intf - powers[i]
        sinr = SINR_lambda(powers[i], interference_at_bs)
        all_sinr.append(sinr)

    return all_sinr
