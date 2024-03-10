"""
https://pythonmatplotlibtips.blogspot.com/2018/11/animation-3d-surface-plot-funcanimation-matplotlib.html
"""

import numpy as np

print("numpy : " + np.version.full_version)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib
from functools import partial
import subprocess

print("matplotlib: " + matplotlib.__version__)


def multivariate_gaussian(x: np.ndarray, y: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """
    generate a multivariate gaussian surface in z direction
    """
    return 1 / np.sqrt(sig) * np.exp(-(x**2 + y**2) / sig**2)


def update_plot(frame_number: int, zarray: np.ndarray, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, zarray[:, :, frame_number], cmap="magma")


N = 150  # meshsize
fps = 10  # frame rate
frn = 50  # frame number of animation
xlims = {"lower": -4, "upper": 4}

# setup equally spaced x axis points
x = np.linspace(xlims["lower"], xlims["upper"], N + 1)
x, y = np.meshgrid(x, x)

zarray = np.zeros((N + 1, N + 1, frn))

# define the function
f = lambda x, y, sig: 1 / np.sqrt(sig) * np.exp(-(x**2 + y**2) / sig**2)
for i in range(frn):
    zarray[:, :, i] = f(x, y, 1.5 + np.sin(i * 2 * np.pi / frn))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plot = [ax.plot_surface(x, y, zarray[:, :, 0], color="0.75", rstride=1, cstride=1)]
ax.set_zlim(0, 1.1)
ani = animation.FuncAnimation(
    fig, update_plot, frn, fargs=(zarray, plot), interval=1000 / fps
)
fn = "plot_surface_animation_funcanimation"
ani.save(fn + ".mp4", writer="ffmpeg", fps=fps)
ani.save(fn + ".gif", writer="imagemagick", fps=fps)
cmd = f"convert {fn}.gif -fuzz 5%% -layers Optimize {fn}_r.gif"
subprocess.check_output(cmd)
