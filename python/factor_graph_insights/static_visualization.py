
import logging
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, Dict, List


class VisualizerBase:
    
    def __init__(self):
        """"""    
    
    
    def plot_pose_covariance(
        singular_values: np.array,
        Determinant: np.array = None,
        adjacency_matrix_data: Dict = None,
        metadata: Dict = None,
        image_ids: np.array = None,
        file_num:int = None,
        save_location:Union[str, Path] = None,
    ):
        """plots the covariance of poses for n dimensional pose
        from the singular values of the marginal covariance matrix.

        Args:
            singular_values (np.array): _description_
        """
        
        # if fig_handles is None:
        #     fig1, fig2, fig3 = None, None, None
        # else:
        #     fig1, fig2, fig3 = fig_handles

        if metadata is not None:
            save_location = Path(metadata['plot_save_location'])
            node_ids = metadata['node_ids']
            image_ids = metadata['image_ids']
            x_ = image_ids[[node_ids]]
            file_num = metadata['file_id']
            
        M, N = singular_values.shape
        labels = ["roll", "pitch", "yaw"]

            
        fig1, ax_theta = plt.subplots(3, 1, figsize=(12, 12))
        for i, ax in enumerate(ax_theta):
            ax.plot(x_, singular_values[:, i], "--*")
            ax.set_xlabel("Pose id ", fontsize=18)
            ax.set_ylabel(f"{labels[i]}", fontsize=18)
            ax.grid(visible=True)
        fig1.suptitle(
            f"Trends of singular Values of covariance matrix - angles - {file_num}", fontsize=18
        )
        
        # Singular values
        fig2, ax_position = plt.subplots(3, 1, figsize=(12, 12))
        labels = ["x(m)", "y(m)", "z(m)"]
        for i, ax in enumerate(ax_position):
            ax.plot(x_, singular_values[:, i + 3], "--*")
            ax.set_xlabel("Pose id ", fontsize=18)
            ax.set_ylabel(f"{labels[i]}", fontsize=18)
            ax.grid(visible=True)
        fig2.suptitle(
            f"Trends of singular Values of covariance matrix - positions - {file_num}]", fontsize=18
        )
        
        # Determinanant
        if Determinant is not None:
            fig3 = plt.figure(figsize=(20, 6))
            gs = fig3.add_gridspec(1, 2, width_ratios=[2, 1])
            ax = [] 
            ax.append(fig3.add_subplot(gs[0]))
            ax.append(fig3.add_subplot(gs[1]))
            ax[0].plot(x_, Determinant, "--*")
            for i, (xi, yi) in enumerate(zip(x_, Determinant)):
                ax[0].annotate(str(i), (xi,yi), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=10, rotation=45)
            ax[0].set_yscale("log")
            ax[0].set_xlabel("Pose id ", fontsize=18)
            ax[0].set_xticks(image_ids[[node_ids]])
            ax[0].xaxis.set_ticks(image_ids[[node_ids]])
            ax[0].tick_params(axis="both", which='major', 
                            labelsize=8, labelrotation=60)
            # ax2 = ax[0].twiny()
            # ax2.set_xlabel("ID in adjacency graph")
            #ax[0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=100))
            ax[0].set_ylabel(f"Determinant of \nrelative marginal covariances of poses \n(Log space)", fontsize=14)
            ax[0].minorticks_on()
            ax[0].grid(visible=True, which='both', linestyle=":")
            ax[0].set_title(f"{metadata['dataset_name']} - {len(node_ids)} keyframes - {int(image_ids[-1].item()/20)}s", fontsize=18)
            
            M = adjacency_matrix_data['M']
            node_ids = node_ids
            ax[1].spy(M[:len(node_ids), :len(node_ids)])
            ax[1].xaxis.set_ticks(node_ids - node_ids[0])
            ax[1].yaxis.set_ticks(node_ids - node_ids[0])
            ax[1].tick_params(axis='both', which='major', labelsize=10)
            ax[1].set_title("Adjaceny graph of the bundle adjustment problem")
            plt.tight_layout()
        if metadata['plot']:
            plt.show()
        
        if save_location is not None:      
            fig1.savefig(save_location.joinpath(f"angle_{file_num}.png"))
            fig2.savefig(save_location.joinpath(f"position_{file_num}.png"))
            fig3.savefig(save_location.joinpath(f"{metadata['dataset_name']}.pdf"))
            logging.info("Saving plots")


def plot_pose_covariance(
    singular_values: np.array,
    Determinant: np.array = None,
    image_ids: np.array = None,
    file_num:int = None,
    save_location:Union[str, Path] = None,
    pause:bool = False,
):
    """plots the covariance of poses for n dimensional pose
       from the singular values of the marginal covariance matrix.

    Args:
        singular_values (np.array): _description_
    """
    
    # if fig_handles is None:
    #     fig1, fig2, fig3 = None, None, None
    # else:
    #     fig1, fig2, fig3 = fig_handles

    if isinstance(save_location, str):
        save_location = Path(save_location)
    
    M, N = singular_values.shape
    if image_ids is not None:
        x_ = image_ids    
    else:
        x_ = range(0, M)
    labels = ["roll", "pitch", "yaw"]

        
    fig1, ax_theta = plt.subplots(3, 1, figsize=(12, 12))
    for i, ax in enumerate(ax_theta):
        ax.plot(x_, singular_values[:, i], "--*")
        ax.set_xlabel("Pose id ", fontsize=18)
        ax.set_ylabel(f"{labels[i]}", fontsize=18)
        ax.grid(visible=True)
    fig1.suptitle(
        f"Trends of singular Values of covariance matrix - angles - {file_num}", fontsize=18
    )
    
    # Singular values
    fig2, ax_position = plt.subplots(3, 1, figsize=(12, 12))
    labels = ["x(m)", "y(m)", "z(m)"]
    for i, ax in enumerate(ax_position):
        ax.plot(x_, singular_values[:, i + 3], "--*")
        ax.set_xlabel("Pose id ", fontsize=18)
        ax.set_ylabel(f"{labels[i]}", fontsize=18)
        ax.grid(visible=True)
    fig2.suptitle(
        f"Trends of singular Values of covariance matrix - positions - {file_num}]", fontsize=18
    )
    
    # Determinanant
    if Determinant is not None:
        fig3, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.plot(x_, Determinant, "--*")
        ax.set_yscale("log")
        ax.set_xlabel("Pose id ", fontsize=18)
        ax.set_ylabel(f"Determinant of Sigma (Log space)", fontsize=18)
        ax.grid(visible=True)
        fig3.suptitle(f"Trend of determinants of covariance matrix - {file_num}", fontsize=18)
    
    if save_location is not None:      
        fig1.savefig(save_location.joinpath(f"angle_{file_num}.png"))
        fig2.savefig(save_location.joinpath(f"position_{file_num}.png"))
        fig3.savefig(save_location.joinpath(f"det_{file_num}.png"))
        logging.info("Saving plots")
    
    if pause:
            plt.show(block=True)
    else: 
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)
    return [fig1, fig2, fig3]

def visualize_images(image_id_list: List):
    """"""
    

def plot_error_from_groundtruth(image_id_list: List, covariance: List ):
    """"""
