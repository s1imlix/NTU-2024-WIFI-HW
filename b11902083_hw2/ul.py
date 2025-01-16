import matplotlib.pyplot as plt
from helper import *
import matplotlib.gridspec as gridspec

Pt = 23
Gt = 14 # Gain in db
Gr = 14

Ht = 1.5
Hr = 51.5

mg = 20
n = 50 # Number of devices
isd = 500
R = isd / math.sqrt(3) # Radius of hexagon
N = 1 # reuse factor
D = math.sqrt(3 * N) * R
T = 300 # Kelvin
B = 10**7 # Hz

fig = plt.figure(figsize=(16, 8))
fig.suptitle("Problem 2: Uplink Power and SINR for central cell")
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

# ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))

ax1.grid(True)

device_distance = []
device_power = []
print("Subproblem 1-2")
ax2 = fig.add_subplot(gs[0, 1])
# Plot each device's received power
ax2.set_title(f"Received power (dB) of central BS from device at distance (m)")
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Received power (dB)")
for i in range(n):
    d = np.sqrt(device_x[i]**2 + device_y[i]**2)
    p = two_ray(Pt + Gt + Gr, Ht, Hr, d, True)
    ax2.scatter(d, p, color='b')
    device_distance.append(d)
    device_power.append(p)

ax2.grid(True)


S = sum([un_dB(p) for p in device_power])
print("Subproblem 1-3")
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title(f"SINR (dB) of central BS from device at distance (m)")
ax3.set_xlabel("Distance (m)")
ax3.set_ylabel("SINR (dB)")
for i in range(n):
    p_for_i = un_dB(device_power[i])
    ax3.scatter(device_distance[i], SNIR(p_for_i, T, B, S - p_for_i), color='g')

ax3.grid(True)
plt.tight_layout()
plt.savefig("uplink.png")
