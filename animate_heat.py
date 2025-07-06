import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

files = sorted(glob.glob("heat_*.npy"))
data = [np.fromfile(f, dtype=np.float32).reshape(128, 128) for f in files]

fig, ax = plt.subplots()
im = ax.imshow(data[0], cmap='hot', origin='lower', vmin=0, vmax=np.max(data[-1]))
ax.set_title("Temperature Evolution")

def update(frame):
    im.set_array(data[frame])
    ax.set_title(f"Step {frame*40}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(data), interval=100)
ani.save("heat_evolution.gif", writer='imagemagick')
plt.show()

