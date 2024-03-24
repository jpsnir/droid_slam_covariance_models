import logging
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, Dict, List
from pathlib import Path
import cv2 as cv

class BaseVisualizer:
    """
    Simple visualizer for a trajectory that has the functionality for making trajectory plots.
    """

    def __init__(self, metadata: Dict = None):
        """
        initialize the plot canvas
        """

        self.angular_labels = ["roll", "pitch", "yaw"]
        self.position_labels = ["x(m)", "y(m)", "z(m)"]
        if metadata is not None:
            if metadata["plot_save_location"] is not None:
                self.save_location = Path(metadata["plot_save_location"])
            self.file_num = metadata["file_id"]
            self.show_plot = metadata["plot"]
            self.node_ids = metadata["node_ids"]
            self.image_ids = metadata["image_ids"]
            self.x_ = self.image_ids[[self.node_ids]]
            self.dataset_name = metadata["dataset_name"]

    def plot_trajectory(self, poses: np.array, metadata: dict):
        """
        plot a 3d trajectory.
        """
        from factor_graph_insights.fg_builder import DataConverter
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        print(poses)
        node_ids = metadata["node_ids"]
        poses = poses[[node_ids]]
        ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], "-*")
        ax.plot(
            poses[0, 0],
            poses[0, 1],
            poses[0, 2],
            "rX",
        )

        for i, p in enumerate(poses):
            ax.text(
                poses[i, 0],
                poses[i, 1],
                poses[i, 2],
                f"{i}",
                color="black",
                va="bottom",
            )
            pose_w_c = DataConverter.to_gtsam_pose(poses[i]).inverse()
            T = pose_w_c.matrix()
            self._plot_coordinate_frame(
                ax,
                T[:3, :3],
                origin=[poses[i, 0], poses[i, 1], poses[i, 2]],
                size=0.05)

        ax.text(poses[0, 0],
                poses[0, 1],
                poses[0, 2],
                "S",
                color="red",
                va="bottom")
        ax.text(poses[-1, 0],
                poses[-1, 1],
                poses[-1, 2],
                "E",
                color="green",
                va="bottom")
        if metadata["plot_save_location"] is not None:
            fig.savefig(metadata["plot_save_location"].joinpath(
                f"factor_graph_trajectory.pdf"))

    def _plot_coordinate_frame(self, ax, R, origin=[0, 0, 0], size=1):
        """
        Plot a coordinate frame defined by the rotation matrix R.

        Parameters:
            ax (Axes3D): Matplotlib 3D axis object.
            R (numpy.ndarray): 3x3 rotation matrix.
            origin (list): Origin point of the coordinate frame.
            size (float): Size of the coordinate frame axes.
        """
        axes = size * R
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axes[0, 0],
            axes[1, 0],
            axes[2, 0],
            color="r",
            label="X",
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axes[0, 1],
            axes[1, 1],
            axes[2, 1],
            color="g",
            label="Y",
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axes[0, 2],
            axes[1, 2],
            axes[2, 2],
            color="b",
            label="Z",
        )

    def visualize_images(self, image_paths_list: List):
        """
        """
        fig = plt.figure()
        plt.show(block=False)
        logging.info(f"Number of images to display: {len(image_paths_list)}")
        for img_path in image_paths_list:
            path = str(img_path.absolute()).replace(" ","")
            img = cv.imread(path)
            plt.imshow(img)
            plt.pause(1)
            
    def plot_error_from_groundtruth(image_id_list: List, covariance: List):
        """"""

        raise NotImplementedError


class CovTrendsVisualizer(BaseVisualizer):
    """
    Extends base visualizer to add covariance plots
    """

    def __init__(self, metadata):
        super().__init__(metadata=metadata)
            

    def plot_trends_determinants(self, Determinant,
                                 adjacency_matrix: np.array):
        """ 
        """
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        ax = []
        ax.append(fig.add_subplot(gs[0]))
        ax.append(fig.add_subplot(gs[1]))
        ax[0].plot(self.x_, Determinant, "--*")
        for i, (xi, yi) in enumerate(zip(self.x_, Determinant)):
            ax[0].annotate(
                str(i),
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
                rotation=45,
            )
        ax[0].set_yscale("log")
        ax[0].set_xlabel("Pose id ", fontsize=18)
        # breakpoint()
        ax[0].set_xticks(self.x_)
        ax[0].xaxis.set_ticks(self.x_)
        ax[0].tick_params(axis="both",
                          which="major",
                          labelsize=8,
                          labelrotation=60)
        ax[0].set_ylabel(
            f"Determinant of \nrelative marginal covariances of poses \n(Log space)",
            fontsize=14,
        )
        ax[0].minorticks_on()
        ax[0].grid(visible=True, which="both", linestyle=":")
        ax[0].set_title(
            f"{self.dataset_name} - {len(self.node_ids)} keyframes\
            - {int(self.image_ids[-1].item()/20)}s",
            fontsize=18,
        )

        M = adjacency_matrix
        node_ids = self.node_ids
        ax[1].spy(M[:len(node_ids), :len(node_ids)])
        ax[1].xaxis.set_ticks(node_ids - node_ids[0])
        ax[1].yaxis.set_ticks(node_ids - node_ids[0])
        ax[1].tick_params(axis="both", which="major", labelsize=10)
        ax[1].set_title("Adjaceny graph of the bundle adjustment problem")
        plt.tight_layout()

        if self.save_location is not None:
            fig.savefig(
                self.save_location.joinpath(
                    f"{self.dataset_name}.pdf"))
            logging.info("Saved plots of determinants and adjacency matrices")

        if self.show_plot:
            plt.show()

    def plot_trends_singular_values(self, singular_values: np.array):
        """
        plt the trends of singular values
        """
        M, N = singular_values.shape
        fig1, ax_theta = plt.subplots(3, 1, figsize=(12, 12))
        for i, ax in enumerate(ax_theta):
            ax.plot(self.x_, singular_values[:, i], "--*")
            ax.set_xlabel("Pose id ", fontsize=18)
            ax.set_ylabel(f"{self.angular_labels[i]}", fontsize=18)
            ax.grid(visible=True)
        fig1.suptitle(
            f"Trends of singular Values of covariance matrix - angles",
            fontsize=18,
        )

        # Singular values
        fig2, ax_position = plt.subplots(3, 1, figsize=(12, 12))
        for i, ax in enumerate(ax_position):
            ax.plot(self.x_, singular_values[:, i + 3], "--*")
            ax.set_xlabel("Pose id ", fontsize=18)
            ax.set_ylabel(f"{self.position_labels[i]}", fontsize=18)
            ax.grid(visible=True)
        fig2.suptitle(
            f"Trends of singular Values of covariance matrix - positions",
            fontsize=18,
        )

        if self.save_location is not None:
            fig1.savefig(
                self.save_location.joinpath(f"angle_{self.file_num}.png"))
            fig2.savefig(
                self.save_location.joinpath(f"position_{self.file_num}.png"))
            logging.info("Saving plots of singular values")

        if self.show_plot:
            plt.show()


class BaseAnimator:
    """
    provides the basic functionality of animation using
    matplotlib to update plots and graphs, given  the data.
    """

    def __init__(self, args):
        """ 
        """
        

    def generator():
        """ 
        """
        
    def animate():
        """
        """
        
    def save():
        """ """


class CovarianceResultsAnimator(BaseAnimator):

    def __init__(self, args):
        """ """
        # 1. canvas
        # 2. image data,
        # 3. single factor graph
        # 4.

    def animate():
        """ """

    def save():
        """ """
