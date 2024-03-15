import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Draw the graph
pos = nx.spring_layout(G)  # Positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=200, node_color="skyblue", font_size=10, font_weight="bold", ax=ax[0])
ax[0].set_title("(A): Covisibility graph")

ax[1].spy(adj_matrix)
# ax[1].xaxis.set_ticks(range(0,10))
# ax[1].yaxis.set_ticks(range(0,10))
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].set_title("(B): Corresponding adjacency graph")
# Display the plot
plt.show()