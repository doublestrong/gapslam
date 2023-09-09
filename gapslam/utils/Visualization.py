import random

import gtsam
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from geometry.TwoDimension import SE2Pose
from slam.Variables import Variable, VariableType, R2Variable, SE2Variable
from factors.Factors import PriorFactor, Factor, BinaryFactor, KWayFactor, BinaryFactorWithNullHypo
from typing import Dict, Tuple, List, Union, Iterable
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import open3d as o3d
import scipy

def plot_pose(ax, pose, marker_size = 40, color='red',**kwargs):
    marker = mpl.markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
    marker._transform = marker.get_transform().rotate_deg(90 + pose.theta * 180 / np.pi)
    ax.scatter((pose.x), (pose.y), marker=marker,
               s=marker_size, c = color, **kwargs)

def plot_point(ax, point, marker_size = 40, color='blue', label:str = None, label_offset = (3,3),**kwargs):
    marker = "*"
    x, y = point.x, point.y
    ax.scatter((x), (y), marker=marker,
               s=marker_size, c = color, **kwargs)
    if label is not None:
        ax.text(x + label_offset[0], y + label_offset[1],
                s=label)

def plot_likelihood_factor(ax, factor, var2truth, width=.5):
    if isinstance(factor, BinaryFactor):
        var1, var2 = factor.vars
        x1, y1 = var2truth[var1][:2]
        x2, y2 = var2truth[var2][:2]
        if isinstance(factor, BinaryFactorWithNullHypo):
            ax.plot([x1, x2], [y1, y2],'--', c='red',
                    linewidth=width,
                    alpha=.5)
        else:
            ax.plot([x1, x2], [y1, y2], c='black', linewidth=width)
    elif isinstance(factor, KWayFactor):
        var1 = factor.root_var
        for var2 in factor.child_vars:
            x1, y1 = var2truth[var1][:2]
            x2, y2 = var2truth[var2][:2]
            ax.plot([x1, x2], [y1, y2], '--',
                    c='black',
                    linewidth=width,
                    alpha=0.5)
    else:
        raise ValueError("Unknown factor type.")

def plot_2d_samples(
        ax: plt.Axes = None,
        samples_mapping: Dict[Variable, np.ndarray] = None,
        samples_array: np.array = None,
        variable_ordering: List[Variable] = None,
        has_orientation: bool = False,
        colors: Union[List[str], Dict[Variable, str]] = None,
        marker_size: float = None,
        xlabel: str = "x (m)",
        ylabel: str = "y (m)",
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        title: str = None,
        file_name: str = None,
        show_plot: bool = False,
        legend_on: bool = False,
        equal_axis: bool = True,
        truth: Dict[Variable, np.ndarray] = None,
        truth_odometry_color: str = "k",
        truth_landmark_measurement_color: str = "k",
        truth_odometry_linewidth: float = 1,
        truth_landmark_measurement_linewidth: float = 1,
        truth_pose_marker: str = "*",
        truth_landmark_marker: str = "*",
        truth_pose_markersize: float = 15,
        truth_landmark_markersize: float = 15,
        truth_pose_color: str = "r",
        truth_landmark_color: str = "b",
        truth_factors: Iterable[Factor] = None,
        truth_label_offset: Tuple[float, float] = (0, -4),
        plot_all_meas: bool = True,
        plot_meas_give_pose: Iterable[Variable] = None,
        rbt_traj_no_samples = False,
        rbt_traj_color = "r",
        fig_size = None,
        truth_R2 = True,
        truth_SE2 = True,
        **kwargs
) -> plt.Axes:
    """
    Generate samples scatter plot from either:
        1) a dictionary that maps variable to samples in numpy array
        2) a numpy array containing all samples
    Each row of samples is one instance (x, y, theta)
    :param samples_mapping: the dictionary from variable to samples
    :param samples_array: samples in numpy array form
    :param variable_ordering: when samples are in the dictionary form, it
        specifies the ordering by which samples are plotted; when samples are in
        the numpy array form, it also specifies the ordering by which variables
        are organized in the array
    :param has_orientation
    :param colors: colors of variables; when specified in the list form, the
        variable_ordering must be specified (the elements are associated)
    :param marker_size
    :param xlabel
    :param ylabel
    :param xlim
    :param ylim
    :param title
    :param file_name: when specified, the figure would be saved
    :param show_plot: when True, the figure would be displayed
    :Keyword Arguments:
        All will be applied to the scatter plot function
    """
    # Input processing
    # Convert input into samples_mapping and variables_ordering
    #   colors are also converted into the dictionary form

    if ax is None:
        if fig_size is None:
            plt.figure()
        else:
            plt.figure(figsize=fig_size)
        ax = plt.gca()

    _dim_per_var = 3 if has_orientation else 2
    if (samples_mapping is None) and (
            samples_array is None):
        raise ValueError("The samples must be specified in one of "
                         "dictionary form or array form")
    if (samples_array is not None) and (variable_ordering is None):
        raise ValueError("When samples are entered in the array form, "
                         "the ordering must be specified")
    if colors:
        if isinstance(colors, list):
            if not variable_ordering:
                raise ValueError("The colors can only be specified as a list "
                                 "when variable_ordering is specified")
            elif len(colors) != len(variable_ordering):
                raise ValueError("The lengh of the color list is incorrect")
            else:
                colors = {variable_ordering[i]: colors[i] for i in
                          range(len(variable_ordering))}
        elif samples_mapping:
            if set(samples_mapping.keys()) != set(colors.keys()):
                raise ValueError("The colors mapping has incorrect keys")
        elif len(colors.keys()) != len(variable_ordering):
            raise ValueError("The colors mapping has incorrect number of "
                             "keys")
    if variable_ordering:
        if samples_mapping and set(samples_mapping.keys()
                                   ) != set(variable_ordering):
            raise ValueError("The variables in ordering must match those"
                             " in the mapping")
        elif len(samples_array.shape) != 2 or \
                samples_array.shape[1] != np.sum([v.dim for v in variable_ordering]):
            raise ValueError("The number of columns in the samples array must "
                             "match the number of variables in the ordering")
    else:
        # variable_ordering = sorted(samples_mapping.keys())
        variable_ordering = list(samples_mapping.keys())[::-1]
    if not samples_mapping:
        samples_mapping = {}
        for i, var in enumerate(variable_ordering):
            if i == 0:
                start_idx = 0
            else:
                start_idx = np.sum([variable_ordering[k].dim for k in range(i)])
            samples_mapping[var] = samples_array[:, start_idx:start_idx+var.dim]
    if not colors:
        colors = {var: tuple(np.random.random(3)) for var in variable_ordering}

    # Plotting
    marker = mpl.markers.MarkerStyle(marker='^') if has_orientation else \
        mpl.markers.MarkerStyle(marker='.')
    if not marker_size:
        marker_size = 10 if has_orientation else 1

    if rbt_traj_no_samples:
        var_scatter = [var for var in variable_ordering if var.type == VariableType.Landmark]
        rbt_vars = [var for var in variable_ordering if var.type == VariableType.Pose]
        x_list = []
        y_list = []
        # x, y, th = .0, .0, .0
        for var in rbt_vars:
            cur_sample = samples_mapping[var]
            x = np.mean(cur_sample[:, 0])
            y = np.mean(cur_sample[:, 1])
            # th = scipy.stats.circmean(cur_sample[:, 2])
            x_list.append(x)
            y_list.append(y)
        ax.plot(x_list, y_list, c=rbt_traj_color)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        # marker = mpl.markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
        # marker._transform = marker.get_transform().rotate_deg(90 + th * 180 / np.pi)
        # ax.scatter((x), (y), marker=marker,
        #            s=marker_size, c='green', **kwargs)

    else:
        var_scatter = variable_ordering
    # plot samples
    if has_orientation:
        for var in var_scatter:
            cur_samples = samples_mapping[var]
            for i in range(cur_samples.shape[0]):
                if len(cur_samples[i, :]) == 3:
                    x, y, th = cur_samples[i, :]
                    marker = mpl.markers.MarkerStyle(marker='$\u2193$')#Downwards arrow in Unicode: ↓
                    marker._transform = marker.get_transform().rotate_deg(90+th * 180 / np.pi)
                    ax.scatter((x), (y), marker=marker,
                               s=marker_size, c=[colors[var]], **kwargs)
                elif len(cur_samples[i, :]) == 2:
                    x, y = cur_samples[i, :]
                    marker = mpl.markers.MarkerStyle(marker='.')
                    ax.scatter((x), (y), marker=marker,
                               s=marker_size/10, c=[colors[var]], **kwargs)
                else:
                    raise ValueError("A variable's dim must be 2 or 3")
    else:
        for var in var_scatter:
            cur_samples = samples_mapping[var]
            ax.scatter(cur_samples[:, 0], cur_samples[:, 1], marker=marker,
                       s=marker_size, c=[colors[var]], **kwargs)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if legend_on:
        #warning: this effect may be overwritten by equal_axis
        # you may want to set equal_axis to False to hide the extra points "lines"
        if has_orientation:
            lines = []
            for var in variable_ordering:
                if var.dim == 3:
                    marker = mpl.markers.MarkerStyle(marker='$\u2193$')
                    x, y, th = samples_mapping[var][0, :]
                    marker._transform = marker.get_transform().rotate_deg(90 + th * 180 / np.pi)
                    lines.append(ax.scatter((x), (y), marker=marker,
                               s=marker_size, c=[colors[var]], **kwargs))
                else:
                    x, y = samples_mapping[var][0, :]
                    marker = mpl.markers.MarkerStyle(marker='.')
                    lines.append(ax.scatter((x), (y), marker=marker,
                               s=marker_size / 10, c=[colors[var]], **kwargs))
            ax.legend(lines, [var.name for var in variable_ordering])
        else:
            ax.legend([var.name for var in variable_ordering])

    if title is not None:
        ax.set_title(title)

    if equal_axis:
        # warning: this may show extra points drawn for legend on
        ax.axis("equal")

    # Plot ground truth
    if truth is not None:
        nodes = truth.keys()
        for node in nodes:
            if node.type == VariableType.Pose:
                color = truth_pose_color
                marker_size = truth_pose_markersize
                marker = truth_pose_marker
            elif node.type == VariableType.Landmark:
                color = truth_landmark_color
                marker_size = truth_landmark_markersize
                marker = truth_landmark_marker
            else:
                raise ValueError("There exits unsupported variable type")
            if isinstance(node, R2Variable):
                if truth_R2:
                    x, y = truth[node]
                    ax.plot([x], [y], c=color, markersize=marker_size,
                            marker=marker)
                    ax.text(x + truth_label_offset[0], y + truth_label_offset[1],
                            s=node.name, size="x-small")
            elif isinstance(node, SE2Variable):
                if truth_SE2:
                    x, y, theta = truth[node]
                    SE2_marker = mpl.markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
                    SE2_marker._transform = SE2_marker.get_transform().rotate_deg(90 + theta * 180 / np.pi)
                    ax.scatter((x), (y), c=color, marker=SE2_marker,
                               s=marker_size*3)
                    # ax.plot([x], [y], c=color, markersize=marker_size,
                    #         marker=marker)
                    ax.text(x + truth_label_offset[0], y + truth_label_offset[1],
                            s=node.name)
            else:
                raise NotImplementedError("Add ground truth plot for other "
                                          "dimensions")

    if truth_factors:
        for factor in truth_factors:
            if isinstance(factor, PriorFactor):
                continue
            elif isinstance(factor, BinaryFactor):
                var1, var2 = factor.vars
                x1, y1 = truth[var1][:2]
                x2, y2 = truth[var2][:2]
                odom_meas = False
                if (var1.type == VariableType.Pose and
                        var2.type == VariableType.Pose):
                    color = truth_odometry_color
                    width = truth_odometry_linewidth
                    odom_meas = True
                elif ((var1.type == VariableType.Landmark and
                       var2.type == VariableType.Pose) or
                      (var1.type == VariableType.Pose and
                       var2.type == VariableType.Landmark)):
                    color = truth_landmark_measurement_color
                    width = truth_landmark_measurement_linewidth
                else:
                    raise ValueError("Connecting the two types of variables is"
                                     "not supported")
                plot_flag = False
                if plot_all_meas or odom_meas:
                    plot_flag = True
                elif plot_meas_give_pose is not None:
                    if set(factor.vars).intersection(plot_meas_give_pose):
                        plot_flag = True
                if plot_flag:
                    if isinstance(factor, BinaryFactorWithNullHypo):
                        ax.plot([x1, x2], [y1, y2], c='red', linewidth=width)
                    else:
                        ax.plot([x1, x2], [y1, y2], c=color, linewidth=width)
            elif isinstance(factor, KWayFactor):
                var1 = factor.root_var
                plot_flag = False
                if plot_all_meas:
                    plot_flag = True
                elif plot_meas_give_pose is not None:
                    if var1 in set(plot_meas_give_pose):
                        plot_flag = True
                if plot_flag:
                    for var2 in factor.child_vars:
                        x1, y1 = truth[var1][:2]
                        x2, y2 = truth[var2][:2]
                        if (var1.type == VariableType.Pose and
                                var2.type == VariableType.Pose):
                            color = truth_odometry_color
                            width = truth_odometry_linewidth
                        elif ((var1.type == VariableType.Landmark and
                               var2.type == VariableType.Pose) or
                              (var1.type == VariableType.Pose and
                               var2.type == VariableType.Landmark)):
                            color = truth_landmark_measurement_color
                            width = truth_landmark_measurement_linewidth
                        else:
                            raise ValueError("Connecting the two types of variables is"
                                             "not supported")
                        ax.plot([x1, x2], [y1, y2],'--', c=color, linewidth=width, alpha=0.5)
            else:
                raise NotImplementedError("Only visualization of binary factors "
                                          "are supported")

    if xlim is not None:
        # plt.xlim(xlim)
        ax.set_xlim(xlim[0], xlim[1])
    # else:
    #     plt.xlim(plt.xlim())
    if ylim is not None:
        # plt.ylim(ylim)
        ax.set_ylim(ylim[0], ylim[1])
    # else:
    #     plt.ylim(plt.ylim())


    # Show and save plot
    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()

    return ax

def plot2d_mean_rbt_only(vars:List[Variable], samples: np.ndarray, title: str = None, xlim=None, ylim=None,
                       if_legend: bool = False, fname = None,ms:int=None,  if_show = False,legend_font=None):
    # list(self._samples.keys())
    sample_dict = {}
    cur_idx = 0
    for var in vars:
        sample_dict[var] = samples[:,cur_idx:cur_idx+var.dim]
        cur_idx = cur_idx+var.dim
    len_var = len(vars)
    x_list = []
    y_list = []
    lmk_list = []
    for i in range(len_var):
        if vars[i]._type == VariableType.Landmark:
            lmk_list.append(vars[i])
        else:
            cur_sample = sample_dict[vars[i]]
            x = np.mean(cur_sample[:, 0])
            y = np.mean(cur_sample[:, 1])
            x_list.append(x)
            y_list.append(y)
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
    plt.plot(x_list, y_list)
    for var in lmk_list:
        cur_sample = sample_dict[var]
        if ms is not None:
            plt.scatter(cur_sample[:, 0], cur_sample[:, 1],s=ms, label=var.name)
        else:
            plt.scatter(cur_sample[:,0],cur_sample[:,1],label=var.name)
    if if_legend:
        if legend_font is not None:
            plt.legend(fontsize=legend_font)
        else:
            plt.legend()

    if title is not None:
        plt.title(title)
    fig_handle = plt.gcf()
    if fname is not None:
        plt.savefig(fname)
    if if_show:
        plt.show()
    return fig_handle

def plot2d_clutter_rbt(vars:List[Variable], samples: np.ndarray, title: str = None, xlim=None, ylim=None,
                       if_legend: bool = False, fname = None,ms:int=None,  if_show = False, traj_num: int = None,
                       draw_ellipse: bool=False, ellipse_itv:int = 200, draw_samples:int = 0):
    # list(self._samples.keys())
    sample_dict = {}
    cur_idx = 0
    for var in vars:
        sample_dict[var] = samples[:,cur_idx:cur_idx+var.dim]
        cur_idx = cur_idx+var.dim

    if traj_num is None:
        traj_num = samples.shape[0]
    elif traj_num > samples.shape[0]:
        raise ValueError("Invalid traj_num ", traj_num)

    len_var = len(vars)
    x_list = []
    y_list = []
    lmk_list = []

    all_x_list = np.zeros((samples.shape[0], 0))
    all_y_list = np.zeros((samples.shape[0], 0))

    rbt_id = -1
    for i in range(len_var):
        if vars[i]._type == VariableType.Landmark:
            lmk_list.append(vars[i])
        else:
            cur_sample = sample_dict[vars[i]]
            x = np.mean(cur_sample[:, 0])
            y = np.mean(cur_sample[:, 1])
            all_x_list = np.hstack((all_x_list, cur_sample[:, 0:1]))
            all_y_list = np.hstack((all_y_list, cur_sample[:, 1:2]))

            rbt_id += 1
            if (draw_ellipse or (draw_samples > 0)) and rbt_id % ellipse_itv == 0:

                if draw_samples > 0:
                    if draw_samples > cur_sample.shape[0]:
                        plt.scatter(cur_sample[:, 0] , cur_sample[:, 1], s = 0.1)
                    else:
                        plt.scatter(cur_sample[:draw_samples, 0] , cur_sample[:draw_samples, 1], s = 0.1)

                tmp_x = np.mean(cur_sample[:, 0])
                tmp_y = np.mean(cur_sample[:, 1])

                plt.scatter(tmp_x,tmp_y,marker='*')

                confidence_ellipse(cur_sample[:, 0],cur_sample[:, 1],plt.gca(),edgecolor='blue')

                # if rbt_id == ellipse_itv:
                #     plt.text(tmp_x - 5, tmp_y-5, s=vars[i].name)
                # else:
                #     plt.text(tmp_x +3 , tmp_y, s=vars[i].name)

            x_list.append(x)
            y_list.append(y)
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)

    for i in range(traj_num):
        downsampling_indices = random.sample(list(range(all_y_list.shape[0])), 1)
        idx = downsampling_indices[0]
        plt.plot(all_x_list[idx,:], all_y_list[idx,:], color="black",linewidth = 0.2)

    # plt.plot(x_list, y_list, color = 'r',linewidth='0.5',alpha='0.5')
    plt.plot(x_list, y_list, color = 'r',linewidth=0.5,alpha=0.8)
    for var in lmk_list:
        cur_sample = sample_dict[var]
        if ms is not None:
            plt.scatter(cur_sample[:, 0], cur_sample[:, 1],s=ms, label=var.name)
        else:
            plt.scatter(cur_sample[:,0],cur_sample[:,1],label=var.name)
    if if_legend:
        plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    if title is not None:
        plt.title(title)
    fig_handle = plt.gcf()
    if fname is not None:
        plt.savefig(fname)
    if if_show:
        plt.show()
    return fig_handle

def confidence_ellipse(x, y, ax, n_std=1.5, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def xycov2ellipse(mean, cov, ax, n_std=1.5, edgecolor='black', plot = True, **kwargs):
    x = mean[0]  # x-position of the center
    y = mean[1]  # y-position of the center
    w, v = np.linalg.eig(cov)
    a = np.sqrt(w[0])*n_std  # radius on the x-axis
    b = np.sqrt(w[1])*n_std  # radius on the y-axis
    t_rot = np.arctan2(v[1, 0], v[0, 0])  # rotation angle

    t = np.linspace(0, 2 * np.pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    # u,v removed to keep the same center location
    R_rot = np.array([[np.cos(t_rot), -np.sin(t_rot)], [np.sin(t_rot), np.cos(t_rot)]])
    # 2-D rotation matrix

    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
    if plot:
        ax.scatter(x, y, marker='o', c=edgecolor)  # initial ellipse
        ax.plot(x + Ell_rot[0, :], y + Ell_rot[1, :], edgecolor, **kwargs)
        return ax
    else:
        return x + Ell_rot[0, :], y + Ell_rot[1, :]

# def SE2cov2ellipsoidpts(xyt, cov, n_std=1.5, resolution=200):
#     w, v = np.linalg.eig(cov)
#     a = np.sqrt(w[0])*n_std  # radius on the x-axis
#     b = np.sqrt(w[1])*n_std  # radius on the y-axis
#     rot_mat = v
#
#     center_pose = SE2Pose(*xyt)
#
#     az = np.linspace(0, 2 * np.pi, resolution) # azimuth angles
#
#     xpts = a * np.cos(az)
#     ypts = b * np.sin(az)
#     zpts = np.zeros_like(xpts)
#
#     xyzpts = np.hstack((xpts.reshape((-1, 1)),  ypts.reshape((-1, 1)), zpts.reshape((-1, 1))))
#     xyzpts = (rot_mat @ xyzpts.T).T
#     SE2pts = np.array([(center_pose * SE2Pose.by_exp_map(pt)).array[:2] for pt in xyzpts])
#     return SE2pts

def SE2cov2ellipsoidpts(xyt, cov, n_std=1.5, resolution=200):
    w, v = np.linalg.eig(cov)
    a = np.sqrt(w[0])*n_std  # radius on the x-axis
    b = np.sqrt(w[1])*n_std  # radius on the y-axis
    rot_mat = v

    center_pose = gtsam.Pose2(*xyt)

    az = np.linspace(0, 2 * np.pi, resolution) # azimuth angles

    xpts = a * np.cos(az)
    ypts = b * np.sin(az)
    zpts = np.zeros_like(xpts)

    xyzpts = np.hstack((xpts.reshape((-1, 1)),  ypts.reshape((-1, 1)), zpts.reshape((-1, 1))))
    xyzpts = (rot_mat @ xyzpts.T).T
    SE2pts = np.zeros((resolution, 2))
    for i, pt in enumerate(xyzpts):
        tmp_pose = center_pose * gtsam.Pose2.Expmap(pt)
        SE2pts[i] = tmp_pose.x(), tmp_pose.y()
    return SE2pts

def xyzcov2ellipsoidpts(center, cov, n_std=1.5, resolution=20):
    w, v = np.linalg.eig(cov)
    a = np.sqrt(w[0])*n_std  # radius on the x-axis
    b = np.sqrt(w[1])*n_std  # radius on the y-axis
    c = np.sqrt(w[2])*n_std  # radius on the z-axis
    rot_mat = v

    az = np.linspace(0, 2 * np.pi, resolution) # azimuth angles
    el = np.linspace(0, np.pi, int(resolution/2)) # elevation angles

    xpts = a * np.outer(np.cos(az), np.sin(el))
    ypts = b * np.outer(np.sin(az), np.sin(el))
    zpts = c * np.outer(np.ones_like(az), np.cos(el))

    xyzpts = np.hstack((xpts.reshape((-1, 1)),  ypts.reshape((-1, 1)), zpts.reshape((-1, 1))))
    xyzpts = (rot_mat @ xyzpts.T).T + center
    return xyzpts, az, el

def xyzcov2ellipsoid(center, cov, n_std=1.5, edgecolor=[0,0,1], resolution=20, **kwargs):
    """
    :param mean: x, y, z of the center
    :param cov: entry orders x, y, z
    :param n_std: scale of standard deviations
    :param edgecolor: color or lines
    :param kwargs:
    :return: lineset for open3d
    """
    xyzpts, az, el = xyzcov2ellipsoidpts(center, cov, n_std, resolution)
    # lines parallel to x-y
    lines = [[i + j*len(az), i + 1 + j*len(az)] for i in range(len(az)-1) for j in range(len(el))]
    # latitude lines
    lines += [[i + (j-1)*len(az), i + j*len(az)] for i in range(len(az)) for j in range(1, len(el))]
    colors = [edgecolor]*len(lines)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(xyzpts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set