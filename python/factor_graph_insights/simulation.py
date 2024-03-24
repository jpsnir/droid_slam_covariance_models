from __future__ import print_function

import math

import gtsam
import numpy as np
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from typing import List, Union, Dict, Tuple
# from gtsam.symbol_shorthand import X, L
import argparse
import networkx as nx
import logging

'''
Goal is to create a pose graph simulation by creating 
a factor graph from a covisibility matrix.
'''

odom_noise_vals = gtsam.Point3(0.1, 0.1, 0.01)
sigma_error = 0.1 # m
lin_vel = 1 #m/s
ang_vel = np.pi/10 #rad/s
dt = 1 # seconds
n_circle = 2*np.pi/ang_vel

def animate(adj_matrix_pair:np.array):
    # Create a graph
    G = nx.Graph()
    M, N = adj_matrix.shape
    fig = plt.figure(constrained_layout=True, figsize=(15, 15))
    spec = gridspec.GridSpec(nrows=2, ncols=4, figure=fig)
    ax1 = []
    ax1.append(fig.add_subplot(spec[0, 0]))
    ax1.append(fig.add_subplot(spec[0, 1]))
    ax1.append(fig.add_subplot(spec[0, 2]))
    
    ax2 = []
    ax2.append(fig.add_subplot(spec[1, 0]))
    ax2.append(fig.add_subplot(spec[1, 1]))
    ax2.append(fig.add_subplot(spec[1, 2]))
    
    ax3 = fig.add_subplot(spec[:, 2])

    # Add some nodes
    G.add_nodes_from(range(M))

    # Create an empty plot
    fig, ax = plt.subplots()

    # Function to update the graph with new edges


    def edges(adj_matrix_pair: Tuple):
        
        adj_matrix1, adj_matrix2 = adj_matrix_pair
        M, N = adj_matrix1.shape
        for i in range(1, M):
            
            temp1 = adj_matrix1[:i, :i]
            temp2 = adj_matrix2[:i, :i]
            temp_new1 = adj_matrix1[:i + 1, :i + 1]
            temp_new2 = adj_matrix1[:i + 1, :i + 1]
            
            m1 = np.zeros([i + 1, i + 1])
            m2 = np.zeros([i + 1, i + 1])
            m1[:i, :i] = temp1
            m2[:i, :i] = temp2
            rows1, cols1 = np.where(temp_new1 - m1 == 1)
            rows2, cols2 = np.where(temp_new2 - m2 == 1)
            edges1 = zip(rows1.tolist(), cols1.tolist())
            edges2 = zip(rows2.tolist(), cols2.tolist())
            yield edges1, edges2


    def update(vals):
        edges1 = vals[0]
        edges2 = vals[1]
        G.add_edges_from(edges)
        # Draw the updated graph
        ax.clear()
        pos = nx.circular_layout(G)
        nx.draw(G, pos, ax=ax, with_labels=True)
        ax.set_title('Frame')


    # Create the animation
    ani = FuncAnimation(fig, update, frames=edges(adj_matrix_pair), interval=1000, repeat=False)

    # Display the animation
    plt.show()
    

def pose_graph_simulation(true_poses: List[np.array], measurements: List[Dict], ax:plt.Axes = None) -> List:
    """Main runner."""
    # Create noise models
    if ax is None:
        fig, ax = plt.subplots()
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.05, 0.05, 0.01))
    # 1. Create a factor graph container and add factors to it
    graph = gtsam.NonlinearFactorGraph()

    # 2a. Add a prior on the first pose, setting it to the origin
    # A prior factor consists of a mean and a noise ODOMETRY_NOISE (covariance matrix)
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), PRIOR_NOISE))

    # 2b. Add odometry factors
    logging.info(f"Number of measurements : {len(measurements)}")
    for m in measurements:
        i, j = m["ids"]
        mean = m["m_mean"]
        print(mean)
        sigma = m["m_sigma"]
        graph.add(gtsam.BetweenFactorPose2(i, j, gtsam.Pose2(mean[0], mean[1], mean[2]), sigma))
        

    print("\nFactor Graph:\n{}".format(graph))  # print

    # 3. Create the data structure to hold the initial_estimate estimate to the
    # solution. For illustrative purposes, these have been deliberately set to incorrect values
    initial_estimate = gtsam.Values()
    np.random.seed(143)
    for i, true_pose in enumerate(true_poses):
        dx = sigma_error*np.random.randn()
        dy = sigma_error*np.random.randn()
        perturbed_pose = gtsam.Pose2(true_pose[0] + dx, true_pose[1] + dy, 0)
        initial_estimate.insert(i, gtsam.Pose2(perturbed_pose))
    #print("\nInitial Estimate:\n{}".format(initial_estimate))  # print

    # 4. Optimize the initial values using a Gauss-Newton nonlinear optimizer
    # The optimizer accepts an optional set of configuration parameters,
    # controlling things like convergence criteria, the type of linear
    # system solver to use, and the amount of information displayed during
    # optimization. We will set a few parameters as a demonstration.
    parameters = gtsam.GaussNewtonParams()

    # Stop iterating once the change in error between steps is less than this value
    parameters.setRelativeErrorTol(1e-5)
    # Do not perform more than N iteration steps
    parameters.setMaxIterations(100)
    # Create the optimizer ...
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)
    # ... and optimize
    result = optimizer.optimize()
    #print("Final Result:\n{}".format(result))

    # 5. Calculate and print marginal covariances for all variables
    marginals = gtsam.Marginals(graph, result)
    N = len(true_poses)
    # for i in range(1, N):
    #     print("X{} covariance:\n{}\n".format(i,
    #                                          marginals.marginalCovariance(i)))
    D_list = []
    # initial poses
    for i in range(1, N):
        gtsam_plot.plot_pose2_on_axes(ax, initial_estimate.atPose2(i))
    
    for i in range(1, N):
        cov = marginals.marginalCovariance(i)
        gtsam_plot.plot_pose2_on_axes(ax, result.atPose2(i), 0.5,
                              cov)
        D = np.linalg.det(cov)
        D_list.append(D)
    
    ax.grid(visible=True)
    # ax.set_xlim((-20, 20))
    # ax.set_ylim((-20, 20))
    print(D_list)
    
    return D_list

def diagonal_adjacency_graph(matrix_size:int = 10, window_size:int = 4) -> np.array:
    adj_matrix = np.zeros((matrix_size, matrix_size))
    # start from 1 as we dont want to create self loop
    for offset in range(1, window_size+1):
        ones = np.ones(matrix_size - offset)
        adj_matrix += np.diag(ones, k = offset)
    return adj_matrix

def visualize_adjacency_matrix(adjacency_matrix:np.array, weights:np.array = None,
                               title:str = None, axis:plt.Axes = None):
    """
    """
    if axis is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    else:
        ax = axis
        
    if weights is None:
        weights = adjacency_matrix
    M, N = adjacency_matrix.shape
    G = nx.from_numpy_array(adjacency_matrix)
    
    # weights
    G_w = nx.from_numpy_array(weights)
    edge_weights = nx.get_edge_attributes(G, "weight")
    edges, weights = zip(*edge_weights.items())
    scaled_weights = [w*5 for w in weights]
    
    # plotting the matrix and the graph
    
    pos = nx.circular_layout(G)  # Positions for all nodes
    nx.draw(G_w, pos, with_labels=True, node_size=200, node_color="skyblue", 
            font_size=8, font_weight="bold", ax=ax[0])
    ax[1].spy(adjacency_matrix)
    ax[1].xaxis.set_ticks(range(0, M))
    ax[1].yaxis.set_ticks(range(0, N))
    ax[1].tick_params(axis="both",
                          which="major",
                          labelsize=8,
                          labelrotation=60)
    # ax[1].tick_params(axis='x', which='major', labelsize=12)
    ax[1].set_title(f"(B): Corresponding adjacency graph")
    plt.tight_layout()
    # Display the plot

def offdiagonal_adjacency_graph(matrix_size:int = 10, window_size:int = 4, off_diagonal_range: List = None, sd: int = 143) -> np.array:
    """
    create a covisibility graph with off diagonal structure
    """
    if off_diagonal_range is None:
        off_diagonal_range = [0, matrix_size]
        
    # get diagonal adjaceny matrix
    adj_matrix = diagonal_adjacency_graph(matrix_size, window_size)
    
    # enter the offdiagonal elements
    np.random.seed(seed=sd)
    for row in off_diagonal_range:
        # we cannot give the window size
        if (row + window_size < matrix_size):
            col = np.random.randint(row + window_size , matrix_size, 2)
        for c in col:
            adj_matrix[row][c] = 1
    adj_matrix[0][matrix_size-1] = 1
    return adj_matrix

def sequential_overlap_noise_model(image_i: int, image_j: int, window_size:int, factor:float = 0.1):
    """
    simulates a noise model for overlapping frames i and j
    """
    
    # develop a noise model for frame i and j
    weight = 1 - factor*(image_j - image_i - 1)/window_size 
    # noise should increase as the overlap decreases, therefore 1/alpha
    return gtsam.noiseModel.Diagonal.Sigmas(1/weight*odom_noise_vals), weight

def off_diagonal_overlap_noise_model(conf_min: float = 0.5, conf_max:float = 0.9):
    """
    generates a random noise model simulating some random overlap.
    We can have weak constraints as well
    """
    np.random.seed(143)
    weight = np.random.uniform(conf_min, conf_max)
    return gtsam.noiseModel.Diagonal.Sigmas(1/weight*odom_noise_vals), weight

def generate_sequential_poses(n: int = 10):
    """"""
    true_poses = []
    init_state = [0,0,0]
    prev_state = init_state
    true_poses.append(init_state)
    
    for i in range(1, n):
        curr_state = [0, 0, 0]
        curr_state[2] = prev_state[2] + ang_vel*dt
        curr_state[1] = prev_state[1] + lin_vel*np.sin(prev_state[2])*dt
        curr_state[0] = prev_state[0] + lin_vel*np.cos(prev_state[2])*dt
        #print(f'(x, y, theta) : {curr_state - prev_state}')
         
        true_poses.append(curr_state)
        prev_state = curr_state
    return np.array(true_poses)

def poses_and_measurements_from_covisibility_graph(
        adj_matrix: np.array, window_size:int = 4)-> Tuple[List, List]:
    """
    create true poses and measurements with noise given a covisibility graph.
    """
    
    measurements = []
    odometry_noise = []
    M, N = adj_matrix.shape
    logging.info(f"Number of poses = {M}")
    true_poses = generate_sequential_poses(M)
   
    weights = np.zeros([M, N])
    for i in range(0, M):
        for j in range(0, N):
            if adj_matrix[i][j] == 1:
                #print(f"frame diff : {j - i}")
                measurement_mean = np.array([true_poses[j][0] - true_poses[i][0],
                                             true_poses[j][1] - true_poses[i][1],
                                             0])
                if (j - i <= window_size):
                    # sequential overlap
                    measurement_noise, weight = sequential_overlap_noise_model(image_i=i, image_j=j, window_size=window_size)
                else:
                    # off diagonal
                    measurement_noise, weight = off_diagonal_overlap_noise_model()
                    
                measurements.append({"ids": (i, j),
                                     "m_mean": measurement_mean,
                                     "m_sigma": measurement_noise,
                                     "weight": weight})
                weights[i, j] = weight
    #print(measurements)
    return true_poses, measurements, weights
    

def list_of_ints(arg):
    """"""
    return list(map(int, arg.split(',')))


def main():
    ap = argparse.ArgumentParser("Argument parser for simulation.py to simulate relative covariance")
    ap.add_argument("--sim_type", choices=["pose_graph, landmark_pose_graph"], 
                    default = "pose_graph", help="choose which simulation to run")
    ap.add_argument("--connection_type", default = "diagonal", choices=['diagonal', 'offdiagonal','both'], help="covisibility graph simulation")
    ap.add_argument("-N", "--number_nodes", type=int, default = 10, help="number of pose nodes to simulate")
    ap.add_argument("--off_diagonal_frames", type=list_of_ints , help="list of off-diagonal frames")
    ap.add_argument("-w", "--window_size", type=int,  
                    default = 3, help="number of keyframes that overlap in sequential manner")
    args = ap.parse_args()
    
    if args.off_diagonal_frames is None:
        mid = int(args.number_nodes/2)
        off_diagonal_frames = list(range(mid-2, mid + 4))
        logging.info(f"off diagonal frames : {off_diagonal_frames}")
    else:
        off_diagonal_frames = args.off_diagonal_frames
    
    if args.connection_type =="diagonal":
       adj_matrix_list = [diagonal_adjacency_graph(args.number_nodes, args.window_size)]
       title_list = ["diagonal"]
    elif args.connection_type =="offdiagonal":    
        adj_matrix_list = [offdiagonal_adjacency_graph(args.number_nodes,args.window_size, off_diagonal_frames)]
        title_list = ["off-diagonal"]
    elif args.connection_type == "both":
        adj_matrix_list = []
        adj_matrix_list.append(diagonal_adjacency_graph(args.number_nodes, args.window_size))    
        
        adj_matrix_list.append(offdiagonal_adjacency_graph(args.number_nodes,
                                                 args.window_size,
                                                off_diagonal_frames))
        title_list = ["diagonal","off-diagonal"]
        
    title = []
    determinant = {}
    
    #fig1, ax1 = plt.subplots(2, 4, figsize=(15, 15))
    fig = plt.figure(constrained_layout=True, figsize=(15, 15))
    spec = gridspec.GridSpec(nrows=2, ncols=4, figure=fig)
     
    ax3 = fig.add_subplot(spec[:, 3])
    for i, (adj_matrix, title) in enumerate(zip(adj_matrix_list, title_list)):
        ax1 = []
        ax1.append(fig.add_subplot(spec[i, 0]))
        ax1.append(fig.add_subplot(spec[i, 1]))
        ax1.append(fig.add_subplot(spec[i, 2]))
        true_poses, measurements, weights = poses_and_measurements_from_covisibility_graph(adj_matrix=adj_matrix)
        visualize_adjacency_matrix(adjacency_matrix=adj_matrix, weights=weights, title=title, axis=ax1[:2])
        determinant[title] = pose_graph_simulation(true_poses, measurements, ax=ax1[2])
    
    # fig, ax = plt.subplots()
    # for pose in true_poses:
    #     plt.scatter(pose[0], pose[1])
    
    # plt.show()
    
    #print(determinant)
    if args.connection_type=="diagonal":
        ax3.plot(determinant[title_list[0]],'*--r')
        ax3.set_yscale("log")
        ax3.legend(title_list)
        ax3.grid(visible=True)
    elif args.connection_type=="offdiagonal":
        ax3.plot(determinant[title_list[1]],'.--g')
        ax3.set_yscale("log")
        ax3.legend(title_list)
        ax3.grid(visible=True)
    elif args.connection_type=="both":
        ax3.plot(determinant[title_list[0]],'*--r')
        ax3.plot(determinant[title_list[1]],'.--g')
        ax3.set_yscale("log")
        ax3.legend(title_list)
        ax3.grid(visible=True)
    plt.show()
    
if __name__ == "__main__":
    main()