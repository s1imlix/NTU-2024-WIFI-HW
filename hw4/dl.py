import matplotlib.pyplot as plt
from helper import *
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import pandas as pd
import tqdm

Pt = 33
Gt = 14 # Gain in db
Gr = 14

Ht = 51.5
Hr = 1.5

mg = 20
n = 50 # Number of devices
isd = 500
R = isd / math.sqrt(3) # Radius of hexagon
N = 1 # reuse factor
D = math.sqrt(3 * N) * R
T = 300 # Kelvin
B = 10**7 # Hz

coord = [[0,0],[2,-1],[2,-2],[1,1],[1,0],[1,-1],[1,-2],[0,2],[0,1],[2,0],[0,-1],[0,-2],[-1,2],[-1,1],[-1,0],[-1,-1],[-2,2],[-2,1],[-2,0]]
vec_i = np.array([3*R/2, math.sqrt(3)*R/2])
vec_j = np.array([3*R/2, -math.sqrt(3)*R/2])
n_colors = len(coord)

print("1-1: Generating the grid of mobile devices")
plt.figure(figsize=(8, 8))
plt.title("Plot of central BS and 50 MS in central cell")
plt.xlabel("X-coordinate (m)")
plt.ylabel("Y-coordinate (m)")
plt.axis('equal')

x, y = hexagon_vertices(R, 0, 0)
plt.plot(x, y) # Hexagon

plt.scatter(0, 0, color='r', marker='x', s=200) # Base station

device_x, device_y = gen_hexagon(n, R, 0, 0)
plt.scatter(device_x, device_y, color='g') # Mobile devices

# Show legend, blue points are mobile devices, red point is the base station, make it outside the plot
plt.legend(["Central cell", "Base Station", "Mobile Devices"], loc='upper right')

plt.grid(True)
plt.savefig("ms_location.png")
plt.clf()
print("1-1: ms_location.png generated")

print("1-2: Shannon Capacity of each device")

_, _, _, _, base_x, base_y = grid_generator(n, R, coord, vec_i, vec_j)
assert(base_x[0] == 0 and base_y[0] == 0)
B_per_user = B / n
all_distance = [np.sqrt(x**2 + y**2) for x, y in zip(device_x, device_y)]
all_interference = [sum([two_ray(Pt, Ht, Hr, np.sqrt((x - base_x[i])**2 + (y - base_y[i])**2), False) for i in range(1, 19)]) for x, y in zip(device_x, device_y)]
all_sinr_db = [SNIR(two_ray(Pt, Ht, Hr, d, False), T, B_per_user, I) for d, I in zip(all_distance, all_interference)]
C = [B_per_user * np.log2(1 + un_dB(sinr)) for sinr in all_sinr_db]

plt.figure(figsize=(8, 6))
plt.title("Shannon Capacity vs. Distance")
plt.xlabel("Distance (m)")
plt.ylabel("Shannon Capacity (bps)")
plt.scatter(all_distance, C)
plt.grid(True)
plt.savefig("shannon_capacity.png")
plt.clf()
print("1-2: shannon_capacity.png generated")

print("1-3: Bit loss probability of each device")
CBR_param = np.array([0.5, 1, 2]) 
CBR_label = [f'Low ({CBR_param[0]} Mbps)', f'Medium ({CBR_param[1]} Mbps)', f'High ({CBR_param[2]} Mbps)']
CBR_param = CBR_param * 10**6
CBR_param = CBR_param.astype(int)
print("CBR_param:", CBR_param)

miss_rate = []

for CBR in CBR_param:
    miss_rate.append(buffer_simulation_CBR(6*10**6, 1000, CBR, C))
    print('miss_rate for CBR =', CBR, 'is', miss_rate[-1])

plt.figure(figsize=(6, 4))
plt.title("Miss rate vs. Traffic Load")
plt.xlabel("CBR (bps)")
plt.ylabel("Miss rate")

# Plot histogram with only 3 bars corresponding to 3 CBR labels
bars = plt.bar(CBR_label, miss_rate)

# Annotate each bar with the value
for bar in bars:
    yval = bar.get_height()  # Get the height of the bar (its value)
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), 
             ha='center', va='bottom')  # Label with the rounded value above each bar
    
plt.grid(True)
plt.savefig("miss_rate_CBR.png")
plt.clf()
print("1-3: miss_rate.png generated")

print("Bonus: Bit loss probability of each device when Poisson traffic model is used")
miss_rate_poisson = []

for CBR in CBR_param:
    miss_rate_poisson.append(buffer_simulation_Poisson(6*10**6, 1000, CBR, C))
    print('miss_rate for CBR =', CBR, 'is', miss_rate_poisson[-1])

plt.figure(figsize=(6, 4))
plt.title("Miss rate vs. Traffic Load (Poisson)")
plt.xlabel("CBR (bps)")
plt.ylabel("Miss rate")

# Plot histogram with only 3 bars corresponding to 3 CBR labels
bars = plt.bar(CBR_label, miss_rate_poisson)

# Annotate each bar with the value
for bar in bars:
    yval = bar.get_height()  # Get the height of the bar (its value)
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), 
             ha='center', va='bottom')  # Label with the rounded value above each bar

plt.grid(True)
plt.savefig("miss_rate_poisson.png")
plt.clf()
print("Bonus: miss_rate_poisson.png generated")






