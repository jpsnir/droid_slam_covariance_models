import numpy as np
import gtsam
import sys
import time

"""
1. Increase age of graph and get the factor graph data to see if the marginals can be computed for longer distances.
2. Increase the frontend threshold in demo.py to get factor graph data to see if the 
3. Incremental BA problem - adding and removing factors in time and continuously solving along with marginalization. 
4. Provide a prior to the first two poses with the data and solve the factor graphs. Are they all solvable?
"""


class GraphAnalysis:
    """to analyse and compute different aspects of factor graph using gtsam library"""

    def __init__(self, graph: gtsam.NonlinearFactorGraph):
        self._graph = graph

    @property
    def graph(self) -> gtsam.NonlinearFactorGraph:
        return self._graph

    @graph.setter
    def graph(self, g: gtsam.NonlinearFactorGraph):
        self._graph = g

    def factor_graph_linear_system(self):
        """"""

    def marginals(self, i_vals: gtsam.Values):
        """"""
        self._init_values = i_vals
        jac_list = []
        b_list = []
        cov_list = []
        info_list = []
        print(f"Errors init values = {self._graph.error(i_vals)}")
        # lin_graph1 = self._graph.linearize(i_vals)
        # jac, b = lin_graph1.jacobian()
        # info_0 = jac.transpose() @ jac
        # cov_0 = np.linalg.inv(info_0)
        # jac_list.append(jac)
        # b_list.append(b)
        # info_list.append(info)
        # cov_list.append(cov)
        # jac_list.append(jac)
        # b_list.append(b)
        # info_list.append(info)
        # cov_list.append(cov)
        marginals_init = gtsam.Marginals(self._graph, i_vals)
        return marginals_init
        # return info_0, cov_0, marginals_init
        number_of_iters = input("Enter integer number of iterations for optimization:")
        print(f"Number of iterations {number_of_iters}")
        time.sleep(2)
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(int(number_of_iters))
        print(f" LM params: {params}")
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self._graph, self._init_values, params
        )
        result = optimizer.optimize()
        print(f"Final result :\n {result}")
        marginals_new = gtsam.Marginals(self._graph, result)
        flag = input("Linearize graph final values: 0-> No, 1-> yes")
        print(f"Errors final values = {self._graph.error(result)}")
        lin_graph2 = self._graph.linearize(result)
        jac, b = lin_graph2.jacobian()
        info_1 = jac.transpose() @ jac
        cov_1 = np.linalg.inv(info_1)

        return (info_0, info_1), (cov_0, cov_1), (marginals_init, marginals_new)
