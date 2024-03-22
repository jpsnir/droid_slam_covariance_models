from __future__ import print_function

import math

import gtsam
import numpy as np
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Tuple
# from gtsam.symbol_shorthand import X, L
import argparse
import networkx as nx
import logging

'''
Goal is to create a pose graph simulation by creating 
a factor graph from a covisibility matrix.
'''

odom_noise_vals = gtsam.Point3(0.1, 0.1, 0.1)
sigma_error = 0.1 # 1m
lin_vel = 1 #m/s
ang_vel = 0.2 #rad/s

def pose_graph_simulation(true_poses: List[np.array], measurements: List[Dict], ax:plt.Axes = None) -> List:
    """Main runner."""
    # Create noise models
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.03, 0.03, 0.01))
    # 1. Create a factor graph container and add factors to it
    graph = gtsam.NonlinearFactorGraph()

    # 2a. Add a prior on the first pose, setting it to the origin
    # A prior factor consists of a mean and a noise ODOMETRY_NOISE (covariance matrix)
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), PRIOR_NOISE))

    # 2b. Add odometry factors
    
    for m in measurements:
        i, j = m["ids"]
        mean = m["m_mean"]
        sigma = m["m_sigma"]
        graph.add(gtsam.BetweenFactorPose2(i, j, gtsam.Pose2(mean), sigma))
        

    print("\nFactor Graph:\n{}".format(graph))  # print

    # 3. Create the data structure to hold the initial_estimate estimate to the
    # solution. For illustrative purposes, these have been deliberately set to incorrect values
    initial_estimate = gtsam.Values()
    for i, true_pose in enumerate(true_poses):
        dx = sigma_error*np.random.randn()
        dy = sigma_error*np.random.randn()
        perturbed_pose = gtsam.Pose2(true_pose[0] + dx, true_pose[1] + dy, true_pose[2])
        initial_estimate.insert(i, gtsam.Pose2(perturbed_pose))
    print("\nInitial Estimate:\n{}".format(initial_estimate))  # print

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
    print("Final Result:\n{}".format(result))

    # 5. Calculate and print marginal covariances for all variables
    marginals = gtsam.Marginals(graph, result)
    N = len(true_poses)
    for i in range(1, N):
        print("X{} covariance:\n{}\n".format(i,
                                             marginals.marginalCovariance(i)))
    D_list = []
    for i in range(1, N):
        gtsam_plot.plot_pose2(0, initial_estimate.atPose2(i))
    for i in range(1, N):
        cov = marginals.marginalCovariance(i)
        gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5,
                              cov)
        D = np.linalg.det(cov)
        D_list.append(D)
    print(D_list)
    
    plt.axis('equal')
    return D_list

def diagonal_adjacency_graph(matrix_size:int = 10, window_size:int = 4) -> np.array:
    adj_matrix = np.zeros((matrix_size, matrix_size))
    # start from 1 as we dont want to create self loop
    for offset in range(1, window_size+1):
        ones = np.ones(matrix_size - offset)
        adj_matrix += np.diag(ones, k = offset)
    return adj_matrix

def visualize_adjacency_matrix(adjacency_matrix:np.array, weights:np.array = None, title:str = None):
    """
    """
    if weights is None:
        weights = adjacency_matrix
    M, N = adjacency_matrix.shape
    G = nx.from_numpy_array(adj_matrix)
    
    # weights
    G_w = nx.from_numpy_array(weights)
    edge_weights = nx.get_edge_attributes(G, "weight")
    edges, weights = zip(*edge_weights.items())
    scaled_weights = [w*5 for w in weights]
    
    # plotting the matrix and the graph
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    pos = nx.circular_layout(G)  # Positions for all nodes
    nx.draw(G_w, pos, with_labels=True, node_size=200, width=scaled_weights,
            node_color="skyblue", font_size=10, font_weight="bold", ax=ax[0])
    ax[1].spy(adj_matrix)
    ax[1].xaxis.set_ticks(range(0, M))
    ax[1].yaxis.set_ticks(range(0, N))
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    ax[1].set_title(f"(B): Corresponding adjacency graph")
    fig.suptitle(f"{title}")
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
    init_state = np.array([0, 0, 0])
    prev_state = init_state
    true_poses.append(init_state)
    dt = 1 # seconds
    for i in range(1, n):
        curr_state = np.zeros([3, 1])
        curr_state[0] = prev_state[0] + lin_vel*np.cos(prev_state[2])*dt
        curr_state[1] = prev_state[1] + lin_vel*np.sin(prev_state[2])*dt
        curr_state[2] = prev_state[2] + ang_vel*dt 
        true_poses.append(curr_state)
        prev_state = curr_state
    return true_poses

def poses_and_measurements_from_covisibility_graph(
        adj_matrix: np.array, window_size:int = 4)-> Tuple[List, List]:
    """
    create true poses and measurements with noise given a covisibility graph.
    """
    
    measurements = []
    odometry_noise = []
    M, N = adj_matrix.shape
    true_poses = generate_sequential_poses(M)
    weights = np.zeros([M, N])
    for i in range(0, M):
        for j in range(0, N):
            if adj_matrix[i][j] == 1:
                
                measurement_mean = np.array([true_poses[j][0] - true_poses[i][0],
                                             true_poses[j][1] - true_poses[i][1],
                                             true_poses[j][2] - true_poses[i][2]])
                
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
    return true_poses, measurements, weights
    

def list_of_ints(arg):
    """"""
    return list(map(int, arg.split(',')))

if __name__ == "__main__":
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
    
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 12))
    for i, (adj_matrix, title) in enumerate(zip(adj_matrix_list, title_list)):
        true_poses, measurements, weights = poses_and_measurements_from_covisibility_graph(adj_matrix=adj_matrix)
        visualize_adjacency_matrix(adjacency_matrix=adj_matrix, weights=weights, title=title)
        determinant[title] = pose_graph_simulation(true_poses, measurements, ax=ax1[i])
    fig2, ax2 = plt.subplots(1)
    print(determinant)
    ax2.plot(determinant[title_list[0]],'*--r')
    ax2.plot(determinant[title_list[1]],'.--g')
    ax2.set_yscale("log")
    ax2.legend(title_list)
    ax2.grid(visible=True)
    plt.show()