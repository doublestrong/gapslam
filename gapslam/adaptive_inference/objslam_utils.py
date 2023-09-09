from typing import List

import gtsam
import numpy as np
import open3d as o3d
import scipy
from numba import jit

from geometry.ThreeDimension import SE3Pose


class ObjectSegmentsFrame:
    """
    Read object segments in a frame
    """

    def __init__(self, classes: list, scores: list, masks: np.ndarray, img_path:str = None):
        """
        :param classes: a list of n strings of object classes
        :param scores: a list of n object class scores in [0, 1]
        :param masks: a nxwxh matrix of Boolean entries where n is the number of objects, w and w are image width and height
        :return:
        """
        self.obj_cls = classes
        self.scores = scores
        self.masks = np.array(masks)
        self.img_path = img_path


def read_odom(odom_file: str, scale=None):
    arr = np.loadtxt(odom_file, dtype=float)
    kf_id = arr[:, 0].astype(int)
    if scale is None:
        pose_arr = [SE3Pose.by_transQuat(x=row[1], y=row[2], z=row[3], qw=row[7], qx=row[4], qy=row[5], qz=row[6]) for row
                    in arr]
    else:
        pose_arr = [SE3Pose.by_transQuat(x=row[1]*scale, y=row[2]*scale, z=row[3]*scale, qw=row[7], qx=row[4], qy=row[5], qz=row[6]) for row
                    in arr]
    return kf_id, pose_arr


def read_gt(gt_file: str):
    if gt_file != "None":
        arr = np.loadtxt(gt_file, dtype=float)
        pose_arr = [SE3Pose.by_transQuat(x=row[4], y=row[5], z=row[6], qw=row[0], qx=row[1], qy=row[2], qz=row[3]) for
                    row in arr]
        return pose_arr
    else:
        return None



def get_cam_geometries(cam_pose_mats: List[np.ndarray], frustum_vertices,frustum_lines, frustum_colors, traj_color=[0, 1, 0]):
    frustums = []
    cam_points = []
    cam_lines = []
    for i, c_pose in enumerate(cam_pose_mats):
        cam_points.append(c_pose[:3, 3])
        if i > 0:
            cam_lines.append([i - 1, i])
        c_points = (c_pose @ frustum_vertices).T[:, :3]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(c_points),
            lines=o3d.utility.Vector2iVector(frustum_lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(frustum_colors)
        frustums.append(line_set)
    cam_traj = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cam_points),
        lines=o3d.utility.Vector2iVector(cam_lines)
    )
    cam_traj.colors = o3d.utility.Vector3dVector(np.tile(traj_color, (len(cam_lines), 1)))
    return frustums, cam_traj


def objdet2geominfo(obj_seg: ObjectSegmentsFrame):
    obj_box_masks = []
    if len(obj_seg.obj_cls) > 0:
        obj_box_masks = np.zeros((len(obj_seg.obj_cls), obj_seg.masks[0].shape[0], obj_seg.masks[0].shape[1]), dtype=bool)
    obj_yx_mins = []
    obj_yx_maxs = []
    obj_seg_pixels = []
    for obj_id in range(len(obj_seg.obj_cls)):
        obj_seg_pixels.append(np.argwhere(obj_seg.masks[obj_id] == 1))
        seg_pixels = obj_seg_pixels[-1]
        obj_yx_mins.append(np.min(seg_pixels, axis=0))
        obj_yx_maxs.append(np.max(seg_pixels, axis=0))
        y_min, x_min = obj_yx_mins[-1]
        y_max, x_max = obj_yx_maxs[-1]
        # obj_box_masks.append(np.zeros_like(obj_seg.masks[obj_id], dtype=bool))
        obj_box_masks[obj_id][y_min:y_max + 1, x_min:x_max + 1] = True
    return obj_box_masks, obj_yx_mins, obj_yx_maxs, obj_seg_pixels

@jit(nopython=True, cache=True)
def objmasks2geominfo(obj_seg_masks, obj_box_masks):
    obj_yx_mins = []
    obj_yx_maxs = []
    obj_seg_pixels = []
    for obj_id in range(len(obj_seg_masks)):
        obj_seg_pixels.append(np.argwhere(obj_seg_masks[obj_id] == 1))
        seg_pixels = obj_seg_pixels[-1]
        obj_yx_mins.append([np.min(seg_pixels[:, 0]), np.min(seg_pixels[:, 1])])
        obj_yx_maxs.append([np.max(seg_pixels[:, 0]), np.max(seg_pixels[:, 1])])
        y_min, x_min = obj_yx_mins[-1]
        y_max, x_max = obj_yx_maxs[-1]
        # obj_box_masks.append(np.zeros_like(obj_seg.masks[obj_id], dtype=bool))
        obj_box_masks[obj_id][y_min:y_max + 1, x_min:x_max + 1] = True
    return obj_box_masks, obj_yx_mins, obj_yx_maxs, obj_seg_pixels


def objdet2occludedIDs(obj_seg, obj_box_masks, min_relative_ratio, max_occluded_ratio):
    occluded_ids = []
    for obj_i in range(len(obj_seg.obj_cls)):
        for obj_j in range(obj_i + 1, len(obj_seg.obj_cls)):
            intersection = obj_box_masks[obj_i] & obj_box_masks[obj_j]
            inter_sum = np.sum(intersection)
            if inter_sum > 0:
                obj_i_sum = np.sum(obj_seg.masks[obj_i])
                obj_j_sum = np.sum(obj_seg.masks[obj_j])
                # reject detections with small, overlapping bounding boxes
                if obj_i_sum < min_relative_ratio * obj_j_sum:
                    occluded_ids.append(obj_i)
                elif obj_j_sum < min_relative_ratio * obj_i_sum:
                    occluded_ids.append(obj_j)
                # reject the detection which is in the background and has a large intersection area
                obj_i_in_box = np.sum(obj_seg.masks[obj_i] & intersection)
                obj_j_in_box = np.sum(obj_seg.masks[obj_j] & intersection)
                if obj_i_in_box < obj_j_in_box:
                    # background obj id is obj_i
                    if 1.0 * inter_sum / np.sum(obj_box_masks[obj_i]) > max_occluded_ratio and obj_i not in occluded_ids:
                        occluded_ids.append(obj_i)
                else:
                    # background obj id is obj_j
                    if 1.0 * inter_sum / np.sum(obj_box_masks[obj_j]) > max_occluded_ratio and obj_j not in occluded_ids:
                        occluded_ids.append(obj_j)
            # if 3 in occluded_ids:
        # print(occluded_ids)
        # print(occluded_ids)
    return occluded_ids

@jit(nopython=True, cache=True)
def objmasks2occludedIDs(obj_seg_masks, obj_box_masks, min_relative_ratio, max_occluded_ratio):
    occluded_ids = []
    for obj_i in range(len(obj_seg_masks)):
        for obj_j in range(obj_i + 1, len(obj_seg_masks)):
            intersection = obj_box_masks[obj_i] & obj_box_masks[obj_j]
            inter_sum = np.sum(intersection)
            if inter_sum > 0:
                obj_i_sum = np.sum(obj_seg_masks[obj_i])
                obj_j_sum = np.sum(obj_seg_masks[obj_j])
                # reject detections with small, overlapping bounding boxes
                if obj_i_sum < min_relative_ratio * obj_j_sum:
                    occluded_ids.append(obj_i)
                elif obj_j_sum < min_relative_ratio * obj_i_sum:
                    occluded_ids.append(obj_j)
                # reject the detection which is in the background and has a large intersection area
                obj_i_in_box = np.sum(obj_seg_masks[obj_i] & intersection)
                obj_j_in_box = np.sum(obj_seg_masks[obj_j] & intersection)
                if obj_i_in_box < obj_j_in_box:
                    # background obj id is obj_i
                    if 1.0 * inter_sum / np.sum(obj_box_masks[obj_i]) > max_occluded_ratio and obj_i not in occluded_ids:
                        occluded_ids.append(obj_i)
                else:
                    # background obj id is obj_j
                    if 1.0 * inter_sum / np.sum(obj_box_masks[obj_j]) > max_occluded_ratio and obj_j not in occluded_ids:
                        occluded_ids.append(obj_j)
            # if 3 in occluded_ids:
        # print(occluded_ids)
        # print(occluded_ids)
    return occluded_ids

def MahalanobisDistances(possible_lmk_nodes, solver, rbt_var, meas_xy, meas_cov, cur_rbt_mat, cur_rbt_mat_inv, cam_intrinsic, cam_model, lmk2jointCov):
    gs_lmk_pts = solver.vars2points(possible_lmk_nodes)
    lmk_pts = np.hstack((gs_lmk_pts, np.ones((len(gs_lmk_pts), 1))))
    lmk_pts_inCam = (cur_rbt_mat_inv @ lmk_pts.T)[:3, :].T
    lmk_pts_inCam = lmk_pts_inCam / lmk_pts_inCam[:, 2:]
    lmk_pixels = (cam_intrinsic @ lmk_pts_inCam.T)[:2, :].T
    lmk_pixels_delta = lmk_pixels - meas_xy
    cam_noise = gtsam.noiseModel.Gaussian.Covariance(meas_cov)
    mahalanobis_distances = []
    for tmp_i, lmk_var in enumerate(possible_lmk_nodes):
        tmp_f = gtsam.GenericProjectionFactorCal3_S2(
            lmk_pixels[tmp_i], cam_noise, rbt_var.key, lmk_var.key, cam_model)
        tmp_v = gtsam.Values()
        tmp_v.insert(rbt_var.key, gtsam.Pose3(cur_rbt_mat))
        tmp_v.insert(lmk_var.key, gtsam.Point3(*gs_lmk_pts[tmp_i]))
        tmp_hessian = tmp_f.linearize(tmp_v).getA()
        try:
            if lmk_var in lmk2jointCov:
                tmp_joint_cov = lmk2jointCov[lmk_var]
            else:
                tmp_joint_cov = solver.gaussian_solver.get_joint_covariance(vars=[rbt_var.key, lmk_var.key],
                                                                        vartypes=[rbt_var.__class__.__name__,
                                                                                  lmk_var.__class__.__name__])
                lmk2jointCov[lmk_var] = tmp_joint_cov
            da_C = tmp_hessian @ tmp_joint_cov @ tmp_hessian.T + meas_cov
            mahalanobis_distances.append(
                np.dot(lmk_pixels_delta[tmp_i], scipy.linalg.solve(da_C, lmk_pixels_delta[tmp_i], assume_a='pos')))
        except:
            mahalanobis_distances.append(np.inf)
    return mahalanobis_distances

def points2pixels(cur_rbt_mat_inv, xyzpts, cam_intrinsic):
    if len(xyzpts) == 0:
        return []
    else:
        pts_in_cam = (cur_rbt_mat_inv @ np.hstack(
            [xyzpts, np.ones((len(xyzpts), 1))]).T)[:3, :]
        pixels_in_cam = (cam_intrinsic @ (pts_in_cam / pts_in_cam[-1, :]))[:2, :]
        return pixels_in_cam

def read_obj_seg_frame(data_dir: str, frame_id: int, basename: str = ""):
    file_prefix = f"{data_dir}/{basename}{str(frame_id).zfill(5)}-color"
    with open(file_prefix + ".score", "r") as f:
        lines = f.readlines()
        scores = [float(line.strip()) for line in lines]
    with open(file_prefix + ".classes", "r") as f:
        classes = f.readlines()
        classes = [line.strip() for line in classes]
    # scores = np.loadtxt(file_prefix + ".score", dtype=float)
    # classes = np.loadtxt(file_prefix + ".classes", dtype=str)
    masks = np.load(file_prefix + ".masks.npy")
    return ObjectSegmentsFrame(classes, scores, masks, file_prefix + ".png")