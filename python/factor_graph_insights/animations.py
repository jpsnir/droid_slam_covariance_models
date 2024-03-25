import cv2 as cv
import logging
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from typing import Union, Dict, List
from pathlib import Path
import subprocess

class BaseAnimator:
    """
    provides the basic functionality of animation using
    matplotlib to update plots and graphs, given  the data.
    """

    def __init__(self):
        """ 
        """
        self.fig = plt.figure(constrained_layout=True, figsize=(40, 15))
        
    def set_data(self):
        """
        """
        raise NotImplementedError
        

    def generator(self):
        """ 
        """
        raise NotImplementedError
    
    def update(self):
        """
        """
        raise NotImplementedError
    
    def run(self, show=True):
        """
        animate function
        """
        self.ani = FuncAnimation(self.fig, self.update, frames=self.generator, interval=1000, repeat=False)
        if show:
            plt.show()
        else:
            self.save()
        
    def save(self):
        """ 
        """
        self.ani.save(f"./animation" + ".mp4", writer="ffmpeg", fps=1)
        


class CovarianceResultsAnimator(BaseAnimator):

    def __init__(self):
        """ 
        """
        super().__init__()
        spec = gridspec.GridSpec(nrows=2, ncols=3, figure=self.fig)
     
        self.ax_img = self.fig.add_subplot(spec[0, 0])
        self.ax_img.set_title("Keyframe image")
        self.ax_cov = self.fig.add_subplot(spec[0, 1])
        self.ax_cov.set_yscale("log")
        self.ax_cov.set_xlabel("Pose id", fontsize=18)
        self.ax_cov.set_ylabel(
            f"Determinant of \nrelative marginal covariances of poses \n(Log space)",
            fontsize=18)
        self.ax_cov.grid(visible=True, which="both", linestyle=":")
        
        self.ax_traj = self.fig.add_subplot(spec[0, 2], projection="3d")
        self.ax_traj.view_init(elev=80, azim=80, roll = 0)
        self.ax_traj.set_xlabel("x", fontsize=18)
        self.ax_traj.set_ylabel("y", fontsize=18)
        self.ax_traj.set_zlabel("z", fontsize=18)
        self.ax_traj.set_xlim([0, 2])
        self.ax_traj.set_ylim([-1.5, 1.5])
        self.ax_traj.set_zlim([0, 2])
        self.ax_traj.set_box_aspect([2,1,1])

        self.ax_mat = self.fig.add_subplot(spec[1, 1])
        self.ax_mat.set_title("Adjacency graph of the shown V2_03 dataset images")     
        
        # variables
        self.animation_data = None
        self.image_id = None
        self.curr_img = None
        self.curr_pose = None
        self.curr_cov_det = None
        self.adj_matrix = None
    
    def set_data(self, animation_data: Dict):
        """
        """
        self.animation_data = animation_data
        self.ax_cov.set_xlim([0, max(self.animation_data["kf_ids"]+ 20)])
        self.ax_cov.set_ylim([1e-21, 1e-16])
        
    def update(self, id):
        """
        """
        from factor_graph_insights.fg_builder import DataConverter
        self.ax_img.imshow(self.curr_img)
        self.ax_cov.plot(self.image_id, self.curr_cov_det, "--*")
        self.ax_cov.annotate(
            str(id), (self.image_id, self.curr_cov_det), textcoords="offset points",
            xytext=(0, 10), ha="center", fontsize=18, rotation=45,
        )
        self.ax_mat.spy(self.adj_matrix)
        self.ax_traj.plot(self.curr_pose[0], self.curr_pose[1], self.curr_pose[2], "-*")
        self.ax_img.set_title(f"Keyframe {id}",fontsize=18)
        self.ax_traj.set_title(f"Keyframe pose - {id}",fontsize=18)
        pose_w_c = DataConverter.to_gtsam_pose(self.curr_pose).inverse()
        T = pose_w_c.matrix()
        self._plot_coordinate_frame(
                self.ax_traj,
                T[:3, :3],
                origin=[self.curr_pose[0],
                self.curr_pose[1],
                self.curr_pose[2],],
                size=0.15)
        
    def generator(self):
        """
        defines a generator function
        """
        for i, image_id in enumerate(self.animation_data["kf_ids"]):
            path = str(self.animation_data["image_paths"][i].absolute()).replace(" ","")
            self.index = i
            self.curr_img = cv.imread(path)
            self.adj_matrix = self.animation_data["adj_matrix"]
            self.curr_cov_det = self.animation_data["cov_determinants"][i]
            self.curr_pose = self.animation_data["trajectory"][i]
            self.image_id = image_id
            yield i
    
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