import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
# Create an adjacency matrix (example)
adj_matrix = np.array([[0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 1., 1., 1., 0., 1., 0., 0., 0.],
                      [0., 0., 0., 1., 1., 1., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 1., 1., 0., 1.],
                      [0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
)

# Create a graph from the adjacency matrix
G = nx.from_numpy_array(adj_matrix)

fig, ax = plt.subplots(1, 2, figsize=(10, 6))

def spiral_layout(G, start_radius=0, start_theta=0, spacing=1):
    pos = {}
    num_nodes = len(G)
    for i, node in enumerate(G.nodes()):
        radius = start_radius + spacing * i
        theta = start_theta + (2 * math.pi / num_nodes) * i
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        pos[node] = (x, y)
    return pos

pos = spiral_layout(G, start_radius=2, start_theta=math.pi/3, spacing=4)

# Draw the graph
pos = nx.circular_layout(G)  # Positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=200, node_color="skyblue", font_size=10, font_weight="bold", ax=ax[0])
ax[0].set_title("(A): Covisibility graph")

ax[1].spy(adj_matrix)
# ax[1].xaxis.set_ticks(range(0,10))
# ax[1].yaxis.set_ticks(range(0,10))
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].set_title("(B): Corresponding adjacency graph")
plt.tight_layout()
# Display the plot
plt.show()