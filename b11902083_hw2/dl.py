import matplotlib.pyplot as plt
from helper import *
import matplotlib.gridspec as gridspec

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

fig = plt.figure(figsize=(16, 8))
fig.suptitle("Problem 1: Downlink Power and SINR for central cell")
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

print("Subproblem 1-1")
ax1 = fig.add_subplot(gs[:, 0])
ax1.set_aspect('equal')
# Plot showing 50 uniformly random points in a hexagon
ax1.set_title(f"Location of {n} uniformly distributed mobile devices")
ax1.set_xlabel("X-coordinate (m)")
ax1.set_ylabel("Y-coordinate (m)")
ax1.set_xlim(-R-mg, R+mg)
ax1.set_ylim(-R-mg, R+mg)

x, y = hexagon_vertices(R, 0, 0)
ax1.plot(x, y) # Hexagon

ax1.scatter(0, 0, color='r', marker='x', s=200) # Base station

device_x, device_y = gen_hexagon(n, R, 0, 0)
ax1.scatter(device_x, device_y, color='g') # Mobile devices

# Show legend, blue points are mobile devices, red point is the base station, make it outside the plot
ax1.legend(["Central cell", "Base Station", "Mobile Devices"], loc='center left', bbox_to_anchor=(1, 0.5))

ax1.grid(True)


print("Subproblem 1-2")
ax2 = fig.add_subplot(gs[0, 1])
# Plot each device's received power
ax2.set_title(f"Received power of mobile device (dB) at distance (m) from central BS")
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Received power (dB)")
for i in range(n):
    d = np.sqrt(device_x[i]**2 + device_y[i]**2)
    ax2.scatter(d, two_ray(Pt + Gt + Gr, Ht, Hr, d, True), color='b')

ax2.grid(True)
# plt.tight_layout()
# plt.savefig("dl_power.png")


# CCI ver.
"""
print("Subproblem 1-3 CCI")
fig, ax3 = plt.subplots(1, figsize=(8, 8))
ax3.set_title(f"SNIR (dB) of mobile device at distance (m) from central BS")
ax3.set_xlabel("Distance (m)")
ax3.set_ylabel("SNIR (dB)")
for i in range(n):
    d = np.sqrt(device_x[i]**2 + device_y[i]**2)
    I = 2 * (two_ray(Pt + Gt + Gr, Ht, Hr, D, False) + two_ray(Pt + Gt + Gr, Ht, Hr, D + R, False) + two_ray(Pt + Gt + Gr, Ht, Hr, D - R, False)) # CCI
    ax3.scatter(d, SNIR(two_ray(Pt + Gt + Gr, Ht, Hr, d, False), T, B, I), color='b')

ax3.grid(True)
plt.tight_layout()
plt.savefig("dl_SNIR_CCI.png")
"""



# All ver.
coord = [[2,0],[2,-1],[2,-2],[1,1],[1,0],[1,-1],[1,-2],[0,2],[0,1],[0,-1],[0,-2],[-1,2],[-1,1],[-1,0],[-1,-1],[-2,2],[-2,1],[-2,0]] 
# All cells except central cell
vec_i = np.array([3*R/2, math.sqrt(3)*R/2])
vec_j = np.array([3*R/2, -math.sqrt(3)*R/2])

n_colors = len(coord)
xlist, ylist, _, _, base_x, base_y = grid_generator(n, R, coord, vec_i, vec_j)

print("Subproblem 1-3")
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title(f"SINR (dB) of mobile device at distance (m) from central BS")
ax3.set_xlabel("Distance (m)")
ax3.set_ylabel("SINR (dB)")

for i in range(n):
    d = np.sqrt(device_x[i]**2 + device_y[i]**2)
    all_d = [np.sqrt((device_x[i]-base_x[j])**2 + (device_y[i]-base_y[j])**2) for j in range(n_colors)]
    I = sum([two_ray(Pt + Gt + Gr, Ht, Hr, d, False) for d in all_d]) # CCI
    ax3.scatter(d, SNIR(two_ray(Pt + Gt + Gr, Ht, Hr, d, False), T, B, I), color='g')

ax3.grid(True)
plt.tight_layout()
plt.savefig("downlink.png")