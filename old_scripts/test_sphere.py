from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

figure, axis = plt.subplots(1, figsize=(12, 12), dpi=300)
ax = figure.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

elev = 10.0
rot = 90.0 / 180 * np.pi
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='r', linewidth=0, alpha=0.5)

# calculate vectors for "vertical" circle
a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
b = np.array([0, 1, 0])
for rot in [90]:
    rot = rot / 180 * np.pi
    b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (1 - np.cos(rot))
    ax.plot(np.sin(u), np.cos(u), 0, color='k', linestyle='dashed')
    
    vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u), a[2] * np.sin(u) + b[2] * np.cos(u), color='k', linestyle='dashed')
    ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front), b[1] * np.cos(vert_front), a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front),
            color='k')
    
horiz_front = np.linspace(0, np.pi, 100)
ax.plot(np.sin(horiz_front), np.cos(horiz_front), 0, color='k')

ax.view_init(elev=elev, azim=-20)

# Plot z axis and angular momentum vectors #
ax.quiver(0, 0, 0, 0, -1, 0, length=1.2, color='blue', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, 1, length=1.2, color='red', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 1, 0, 0, length=1.2, color='green', arrow_length_ratio=0.1)

# xx, yy = np.meshgrid(range(10), range(10))
# normal = np.array([1, 1, 2])
# point  = np.array([1, 2, 3])
# d = -point.dot(normal)
# z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
# ax.plot_surface(xx, yy, z)


plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
plt.savefig(plots_path + 'TS.png', bbox_inches='tight')
