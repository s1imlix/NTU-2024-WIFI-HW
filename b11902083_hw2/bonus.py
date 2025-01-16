import matplotlib.pyplot as plt
from helper import *
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

Pt = 23
Gt = 14 # Gain in db
Gr = 14

Ht = 1.5
Hr = 51.5

n = 50 # Number of devices
isd = 500
R = isd / math.sqrt(3) # Radius of hexagon
N = 1 # reuse factor
D = math.sqrt(3 * N) * R
T = 300 # Kelvin
B = 10**7 # Hz

coord = [[2,0],[2,-1],[2,-2],[1,1],[1,0],[1,-1],[1,-2],[0,2],[0,1],[0,0],[0,-1],[0,-2],[-1,2],[-1,1],[-1,0],[-1,-1],[-2,2],[-2,1],[-2,0]]
vec_i = np.array([3*R/2, math.sqrt(3)*R/2])
vec_j = np.array([3*R/2, -math.sqrt(3)*R/2])

n_colors = len(coord)
colors = cm.get_cmap('tab20', n_colors)

fig = plt.figure(figsize=(24, 12))
fig.suptitle("Bonus: Uplink Power and SINR for all cells")
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

print("Subproblem 1-1")
ax1 = fig.add_subplot(gs[:, 0])
ax1.set_aspect('equal')
# Plot showing 50 uniformly random points in a hexagon
ax1.set_title(f"Location of {n} uniformly distributed mobile devices on 19 cells")
ax1.set_xlabel("X-coordinate (m)")
ax1.set_ylabel("Y-coordinate (m)")
ax1.set_xlim(-3*isd, 3*isd)
ax1.set_ylim(-3*isd, 3*isd)

xlist, ylist, device_x, device_y, base_x, base_y = grid_generator(n, R, coord, vec_i, vec_j)

for i in range(n_colors):
    i_color = colors(i)
    ax1.plot(xlist[i], ylist[i], color=i_color) # Hexagon 
    ax1.scatter(device_x[i], device_y[i], color=i_color, label=f'Cell ({coord[i][0]}, {coord[i][1]})') # Mobile devices

ax1.scatter(base_x, base_y, color='r', marker='x', label='Base Station', s=100) # Base station

# Show legend, blue points are mobile devices, red point is the base station, make it outside the plot
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))

ax1.grid(True)


print("Subproblem 1-2")
ax2 = fig.add_subplot(gs[0, 1])
# Plot each device's received power
ax2.set_title(f"Received power (dB) of each BS from device at distance (m)")
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Received power (dB)")
for i in range(n_colors):
    all_d = [np.sqrt((device_x[i][j]-base_x[i])**2 + (device_y[i][j]-base_y[i])**2) for j in range(n)]
    all_p = [two_ray(Pt + Gt + Gr, Ht, Hr, d, True) for d in all_d]
    ax2.scatter(all_d, all_p, color=colors(i), label=f'Cell ({coord[i][0]}, {coord[i][1]})')

# ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax2.grid(True)

all_device_x = flatten(device_x)
all_device_y = flatten(device_y)
print("Subproblem 1-3")
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title(f"SINR (dB) of each BS from device at distance (m)")
ax3.set_xlabel("Distance (m)")
ax3.set_ylabel("SINR (dB)")
for i in range(n_colors):
    I=0 # All signal for cell i's BS
    for j in range(len(all_device_x)):
        d = np.sqrt((all_device_x[j]-base_x[i])**2 + (all_device_y[j]-base_y[i])**2)
        I += two_ray(Pt + Gt + Gr, Ht, Hr, d, False)
    
    # Choose the devices within the cell
    cell_all_d = [np.sqrt((device_x[i][j]-base_x[i])**2 + (device_y[i][j]-base_y[i])**2) for j in range(n)] 
    cell_all_p = [two_ray(Pt + Gt + Gr, Ht, Hr, d, False) for d in cell_all_d]
    cell_all_SNIR = [SNIR(p, T, B, I - p) for p in cell_all_p]
    ax3.scatter(cell_all_d, cell_all_SNIR, color=colors(i), label=f'Cell ({coord[i][0]}, {coord[i][1]})')
        
# ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax3.grid(True)
plt.tight_layout()
plt.savefig("bonus.png")

