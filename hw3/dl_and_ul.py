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

coord = [[2,0],[2,-1],[2,-2],[1,1],[1,0],[1,-1],[1,-2],[0,2],[0,1],[0,0],[0,-1],[0,-2],[-1,2],[-1,1],[-1,0],[-1,-1],[-2,2],[-2,1],[-2,0]]
vec_i = np.array([3*R/2, math.sqrt(3)*R/2])
vec_j = np.array([3*R/2, -math.sqrt(3)*R/2])

n_colors = len(coord)

# Sample cell ID
hex_xlist, hex_ylist, _, _, base_x, base_y = grid_generator(n, R, coord, vec_i, vec_j)

plt.figure(figsize=(8, 8))
plt.title("Cell ID configuration")
for i in range(19):
    # Write i+1 as the cell ID at base_x[i], base_y[i]
    plt.plot(hex_xlist[i], hex_ylist[i], 'k-')
    plt.text(base_x[i], base_y[i], str(i+1), fontsize=24, ha='center', va='center')

random_walk_loc_t = random_walk(1, 15, 1, 6, 900, 250, 0)
random_walk_loc_x = [i[0] for i in random_walk_loc_t]
random_walk_loc_y = [i[1] for i in random_walk_loc_t]


plt.savefig("cell_id.png")
plt.clf()
print("cell_id.png generated")


# Actual urban grid
all_grids = urban_generator(n, R, coord, vec_i, vec_j)

plt.figure(figsize=(16, 16)) 
urban_size = len(all_grids)
colors = plt.get_cmap('tab20', urban_size)

all_base_x = flatten([i[2] for i in all_grids])
all_base_y = flatten([i[3] for i in all_grids])

base_id = [i for i in range(19)] * urban_size



all_base_coord = list(zip(all_base_x, all_base_y, base_id))

for i in range(urban_size):
    i_color = colors(i)
    xlist, ylist, base_x, base_y = all_grids[i]
    for j in range(19):
        plt.plot(xlist[j], ylist[j], color=i_color) # hexagon
    base_x_except_central = base_x[:9] + base_x[10:]
    base_y_except_central = base_y[:9] + base_y[10:]
    plt.scatter(base_x_except_central, base_y_except_central, color=i_color, marker='^', label=f'Base station with Central BS {i+1}', s=100) # Base station
    plt.scatter(base_x[9], base_y[9], color=i_color, marker='*', label=f'Central BS {i+1}', s=200) # Central BS

cell_id_y_offset = 100
for i in range(len(all_base_coord)):
    plt.text(all_base_coord[i][0], all_base_coord[i][1] + cell_id_y_offset, str(all_base_coord[i][2] + 1), fontsize=12, ha='center', va='center')

# print(f'Random walk: {random_walk_loc_t}')
# Make line thinner and more opaque
plt.plot(random_walk_loc_x, random_walk_loc_y, color='blue', alpha=0.5, linewidth=4, label='Random walk')
# plt.plot(random_walk_loc_x[0], random_walk_loc_y[0], 'bo', label='Start')
plt.plot(random_walk_loc_x[-1], random_walk_loc_y[-1], 'ro', label='Path End')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
plt.title("Grid with 1 MS's random walk")

plt.savefig("dl_grid_with_walk.png", bbox_inches='tight')
plt.clf()
print("dl_grid_with_walk.png generated")

# Random walk mobility model DL measure
# Setup lambda for two_ray and SINR
two_ray_lambda = lambda d: two_ray(Pt + Gt + Gr, Ht, Hr, d, False)
SINR_lambda = lambda S, I: SNIR(S, T, B, I)

sinr_best_bs = dl_evaluate_handoff(all_base_coord, random_walk_loc_x[0], random_walk_loc_y[0], two_ray_lambda, SINR_lambda)
total_handoff = 0

time_col = []
prev_cell_col = []
new_cell_col = []
for i in range(len(random_walk_loc_t)):
    sinr_current_bs = sinr_best_bs
    sinr_best_bs = dl_evaluate_handoff(all_base_coord, random_walk_loc_x[i], random_walk_loc_y[i], two_ray_lambda, SINR_lambda)
    if sinr_best_bs != sinr_current_bs:
        time_col.append(random_walk_loc_t[i][2])
        prev_cell_col.append(sinr_current_bs + 1)
        new_cell_col.append(sinr_best_bs + 1)
 

df = pd.DataFrame({'Time': time_col, 'Previous Cell': prev_cell_col, 'New Cell': new_cell_col})
df.to_csv('dl_handoff.csv', index=False)
print(f'Total handoff: {len(time_col)}, Handoff data saved to dl_handoff.csv')

# BONUS: Uplink power and SINR
Pt = 23

ms_initial_x = []
ms_initial_y = []
for i in range(100):
    # Uniformly choose from coord
    coord_idx = np.random.randint(0, len(coord))
    center = vec_i*coord[coord_idx][0] + vec_j*coord[coord_idx][1]
    x, y = gen_hexagon(1, R, center[0], center[1])
    ms_initial_x.append(x[0])
    ms_initial_y.append(y[0])

# print(f'MS initial location: {ms_initial_x}, {ms_initial_y}')
ms_inital = zip(ms_initial_x, ms_initial_y)
# Plot location for 100 mobile devices

# Sample cell ID
hex_xlist, hex_ylist, _, _, base_x, base_y = grid_generator(n, R, coord, vec_i, vec_j)
plt.figure(figsize=(8, 8))
for i in range(19):
    plt.plot(hex_xlist[i], hex_ylist[i], 'k-')
    plt.text(base_x[i], base_y[i], str(i+1), fontsize=24, ha='center', va='center')

plt.scatter(ms_initial_x, ms_initial_y, color='g', label='MS', s=20)
plt.legend(loc='best', fontsize=10)
plt.title("Cell ID configuration with 100 MS")

plt.savefig("ul_msinit.png")
plt.clf()
print("ul_msinit.png generated")

# Generate random walk for 100 mobile devices
random_walk_loc_t_all = [random_walk(1, 15, 1, 6, 900, i[0], i[1]) for i in ms_inital]
two_ray_lambda = lambda d: two_ray(Pt + Gt + Gr, Hr, Ht, d, False)

all_ms_bs_id = [-1 for i in range(100)]
all_ms_bs_tmp_id = [-1 for i in range(100)]
all_ms_bs_sinr = [-2**31 for i in range(100)]


time_col = []
ms_col = []
prev_cell_col = []
new_cell_col = []
test_ms = 7
for i in tqdm.tqdm(range(900)):
    # Get all MS coordinates at time i
    ms_x = [j[i][0] for j in random_walk_loc_t_all]
    ms_y = [j[i][1] for j in random_walk_loc_t_all]
    all_ms_coord = list(zip(ms_x, ms_y))
    # Evaluate SINR for each MS based on single BS
    # print(f'Time {i}: {all_ms_bs_id[test_ms] + 1} when location is ({ms_x[test_ms]}, {ms_y[test_ms]})')       
    all_ms_bs_sinr = [-2**31 for i in range(100)] # Reset SINR for each MS
    for bs in all_base_coord:
        bs_x, bs_y, id = bs
        sinr = ul_evaluate_single_bs(all_ms_coord, bs_x, bs_y, two_ray_lambda, SINR_lambda)
        for j in range(100):
            if sinr[j] > all_ms_bs_sinr[j]:
                all_ms_bs_sinr[j] = sinr[j]
                all_ms_bs_tmp_id[j] = id # find the best BS for each MS
    # print(f'Time {i}: {all_ms_bs_tmp_id}')
    # Update BS ID for each MS
    for j in range(100):
        if all_ms_bs_id[j] != all_ms_bs_tmp_id[j]:
            if all_ms_bs_id[j] != -1:
                # print(f'Time {i}: MS {j+1} handoff from BS {all_ms_bs_id[j]+1} to BS {all_ms_bs_tmp_id[j]+1}')
                time_col.append(i)
                ms_col.append(j+1)
                prev_cell_col.append(all_ms_bs_id[j]+1)
                new_cell_col.append(all_ms_bs_tmp_id[j]+1)
            all_ms_bs_id[j] = all_ms_bs_tmp_id[j]

df = pd.DataFrame({'Time': time_col, 'MS': ms_col, 'Previous Cell': prev_cell_col, 'New Cell': new_cell_col})
df.to_csv('ul_handoff.csv', index=False)
print(f'Total handoff: {len(time_col)}, Handoff data saved to ul_handoff.csv')

# Check: MS 1 handoff with plot
plt.figure(figsize=(16, 16))
colors = plt.get_cmap('tab20', urban_size)
for i in range(urban_size):
    i_color = colors(i)
    xlist, ylist, base_x, base_y = all_grids[i]
    for j in range(19):
        plt.plot(xlist[j], ylist[j], color=i_color) # hexagon
    base_x_except_central = base_x[:9] + base_x[10:]
    base_y_except_central = base_y[:9] + base_y[10:]
    plt.scatter(base_x_except_central, base_y_except_central, color=i_color, marker='^', label=f'Base station with Central BS {i+1}', s=100) # Base station
    plt.scatter(base_x[9], base_y[9], color=i_color, marker='*', label=f'Central BS {i+1}', s=200) # Central BS

cell_id_y_offset = 100
for i in range(len(all_base_coord)):
    plt.text(all_base_coord[i][0], all_base_coord[i][1] + cell_id_y_offset, str(all_base_coord[i][2] + 1), fontsize=12, ha='center', va='center')

# print(f'Random walk: {random_walk_loc_t}')
# Make line thinner and more opaque

all_x = [i[0] for i in random_walk_loc_t_all[test_ms]]
all_y = [i[1] for i in random_walk_loc_t_all[test_ms]]
plt.plot(all_x, all_y, color='blue', alpha=0.5, linewidth=4, label=f'Random walk of MS {test_ms}')
plt.plot(all_x[0], all_y[0], 'bo', label='Start')
plt.plot(all_x[-1], all_y[-1], 'ro', label='Path End')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
plt.title("Grid with MS no.1's random walk")

plt.savefig("ul_grid_with_walk.png", bbox_inches='tight')
plt.clf()
print("ul_grid_with_walk.png generated")

















