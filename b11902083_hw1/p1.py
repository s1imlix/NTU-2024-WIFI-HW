import matplotlib.pyplot as plt
from helper import *
from matplotlib.ticker import MaxNLocator

Pt = 33
Gt = 12 # Gain in db
Gr = 12

Ht = 51.5
Hr = 1.5

dmax = 1000 # Length of distance
n = 500 # Number of samples

print("Subproblem 1.1")
fig, [ax1, ax2] = plt.subplots(2,figsize=(10, 10))

ax1.set_title("Received power to distance (Two-ray model)")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Received power (dB)")

shared_data = [two_ray(Pt + Gt + Gr, Ht, Hr, i, True) for i in range(1, dmax, dmax//n)]

ax1.plot([i for i in range(1, dmax, dmax//n)], shared_data)

print("Subproblem 1.2")

ax2.set_title("SNIR to distance (Two-ray model)")
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("SNIR (dB)")

ax2.plot([i for i in range(1, dmax, dmax//n)], [SNIR(un_dB(data), 300, 10**7, 0) for data in shared_data])

ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
ax2.yaxis.set_major_locator(MaxNLocator(nbins=8))

ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
plt.savefig("hw1_p1.png")


