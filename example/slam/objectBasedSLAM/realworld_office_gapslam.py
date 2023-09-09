import json
import os
import time
from typing import List

import gtsam
import numpy as np
import logging

from scipy.stats.distributions import chi2

from evo.core import metrics
from matplotlib import pyplot as plt

from adaptive_inference.objslam_utils import MahalanobisDistances, points2pixels, \
    objmasks2geominfo, objmasks2occludedIDs
from adaptive_inference.GAPSLAM import GAPSLAM
from adaptive_inference.utils import to_Key
from factors.Factors import UnarySE3Factor, SE3RelativeGaussianLikelihoodFactor, CameraProjectionFactor
from geometry.ThreeDimension import SE3Pose
from slam.Variables import SE3Variable, R3Variable, VariableType
import open3d as o3d
import random
from copy import deepcopy
from evo.core.trajectory import PosePath3D
from evo.tools import plot
import yaml
from utils.Visualization import xyzcov2ellipsoidpts


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

def read_obj_seg_frame(data_dir: str, frame_id: int, basename: str = "", admitted_class=None):
    file_prefix = f"{data_dir}/{basename}{frame_id}"
    with open(file_prefix + ".score", "r") as f:
        lines = f.readlines()
        scores = [float(line.strip()) for line in lines]
    with open(file_prefix + ".classes", "r") as f:
        classes = f.readlines()
        classes = [line.strip() for line in classes]
    # scores = np.loadtxt(file_prefix + ".score", dtype=float)
    # classes = np.loadtxt(file_prefix + ".classes", dtype=str)
    masks = np.load(file_prefix + ".masks.npy")
    if admitted_class is not None:
        admitted_entries = []
        for tmp_c in classes:
            if tmp_c in admitted_class:
                admitted_entries.append(True)
            else:
                admitted_entries.append(False)
        classes = [classes[idx] for idx, admitted in enumerate(admitted_entries) if admitted]
        scores = [scores[idx] for idx, admitted in enumerate(admitted_entries) if admitted]
        masks = [masks[idx] for idx, admitted in enumerate(admitted_entries) if admitted]
    return ObjectSegmentsFrame(classes, scores, masks, file_prefix + ".png")

def read_setting(file_path: str):
    with open(file_path, 'r') as stream:
        try:
            from yaml import CLoader as Loader, CDumper as Dumper
        except ImportError:
            from yaml import Loader, Dumper
        yaml_setting = yaml.load(stream, Loader=Loader)
    odom_file = yaml_setting["odom_file"]
    obj_det_dir = yaml_setting["obj_det_dir"]
    img_prefix = yaml_setting["img_prefix"]
    gt_file = yaml_setting["ground_truth_file"]
    output_dir = yaml_setting["output_dir"]
    if img_prefix == "None":
        img_prefix = ""
    logging.info("Settings: {} {} {} {}".format(odom_file, obj_det_dir, img_prefix, gt_file))
    return odom_file, obj_det_dir, img_prefix, gt_file, output_dir


def read_odom(odom_file: str):
    arr = np.loadtxt(odom_file, dtype=float)
    kf_id = arr[:, 0].astype(int)
    pose_arr = [SE3Pose.by_transQuat(x=row[1], y=row[2], z=row[3], qw=row[7], qx=row[4], qy=row[5], qz=row[6]) for row
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

def noise_scale(xyz_scale):
    if xyz_scale < 1.0:
        return xyz_scale
    else:
        return np.log(xyz_scale) + 1

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # file_path = "setting.yaml"
    # odom_file, obj_det_dir, img_prefix, gt_file, output_dir = read_setting(file_path)

    img_prefix=""

    max_occluded_ratio = 0.1
    min_relative_ratio = 0.1

    pixel_range = 150

    data_dir = "realworld_objslam_data"

    admitted_class = ["cup","cereal_box", "trash_can", "skateboard",
                      "office_chair", "football","bottle","traffic_cone","toy_car"]

    std_min_cap = 5

    ablation_setting = {"preprocess": True,
                        "sampleDA": True,
                        "reinit": True}

    rd_seeds = np.arange(1, 2)
    for rd_seed in rd_seeds:
        np.random.seed(rd_seed)
        random.seed(rd_seed)

        perf_dict = {"step_time": [],
                     "pre_process": [],
                     "da": [],
                     "gapslam": [],
                     "GA": [],
                     "sampling": [],
                     "steps": 0,
                     "num_dets": [],
                     "num_lmks": 0,
                     "odom_rmse": .0,
                     "gap_rmse": .0,
                     "reinit": []}

        with open(os.path.join(data_dir, f"cam_params.yaml"), 'r') as stream:
            try:
                from yaml import CLoader as Loader, CDumper as Dumper
            except ImportError:
                from yaml import Loader, Dumper
            cam_setting = yaml.load(stream, Loader=Loader)

        cam_params = cam_setting["intrinsics"]
        cx, cy = cam_params[2], cam_params[3]
        img_w, img_h = cam_setting["img_dim"][0], cam_setting["img_dim"][1]
        cam_model = gtsam.Cal3_S2(cam_params[0], cam_params[1], .0, cam_params[2],
                                  cam_params[3])

        cam_intrinsic = np.array([[cam_params[0], 0, cx],
                                  [0, cam_params[1], cy],
                                  [0, 0, 1]])


        odom_file = f"{data_dir}/kf_traj.txt"
        gt_file = f"{data_dir}/gt_kf_traj.txt"
        obj_det_dir = f"{data_dir}/detic"
        output_dir = f"{data_dir}/results"
        with open(f"{data_dir}/open3d_render_option.json", 'r') as vis_f:
            vis_json = json.load(vis_f)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        run_cnt = 0
        run_dir = os.path.join(output_dir, f"run{run_cnt}")
        while os.path.exists(run_dir):
            run_cnt += 1
            run_dir = os.path.join(output_dir, f"run{run_cnt}")
        os.mkdir(run_dir)

        kf_ids, odom_poses = read_odom(odom_file)
        odom_mats = [tmp_pose.mat for tmp_pose in odom_poses]
        _, gt_poses = read_odom(gt_file)

        translation_scale = 1.0
        trans_noise_scale = translation_scale


        prior_cov = np.array([0.0001]*6)
        orient_std = 0.15 * np.pi / 180
        translation_std = 0.003 * trans_noise_scale

        prior_cov[3:] = (trans_noise_scale**2) * prior_cov[3:]
        prior_cov = np.diag(prior_cov)
        # Rx, Ry, Rz,tx, ty, tz
        odom_cov = np.diag(np.array([orient_std, orient_std, orient_std, translation_std, translation_std, translation_std]) ** 2)

        # print(translation_scale)
        delta_trans_tol = 0.00003 * translation_scale
        delta_rot_tol = .0001 * np.pi/180
        obj_lo_score = 0.6
        if ablation_setting['reinit']:
            prior_cov_scale = 3 * translation_scale
        else:
            # smaller prior cov prevents gtsam::IndeterminantLinearSystemException
            # but results are prone to be affected by initial values
            prior_cov_scale = 1 * translation_scale
        resample_bw = 0.02 * translation_scale
        reinit_tol = 0.03 * translation_scale
        box_pts_min_percent = 0.1


        # bearing factor setting
        min_depth = 0.1 * translation_scale
        max_depth = 4.0 * translation_scale

        eig_threshold = 0.1 * translation_scale

        # data association setting/new landmark
        min_mhl_dist = chi2.ppf(0.99, df=2) # measurements are 2d pixel coordinates

        # sampling setting
        rbt_path_num = 1
        sample_per_path = 500

        run_params = {"scale": translation_scale, "prior_pose_cov": np.diag(prior_cov).tolist(),
                      "odom_pose_cov": np.diag(odom_cov).tolist(), "trans_tol": delta_trans_tol,
                      "rot_tol": delta_rot_tol, "detection_score": obj_lo_score,
                      "artificial_prior_cov": prior_cov_scale, "resample_bw": resample_bw,
                      "reinit_tol": reinit_tol, "min_da_pts_prct": box_pts_min_percent,
                      "min_max_range": [min_depth, max_depth], "min_unimodal_eig": eig_threshold,
                      "min_mhl_dist": float(min_mhl_dist), "rbt_path_num": rbt_path_num,
                      "sample_per_path": sample_per_path}

        run_params = {**run_params, **ablation_setting}

        with open(os.path.join(run_dir, "run_params.yaml"), "w+") as out_stream:
            yaml.dump(run_params, out_stream)

        solver = GAPSLAM(
            bw_method=resample_bw,
            prior_cov_scale=prior_cov_scale,
            reinit_tol=reinit_tol,
            lmk_sample_num=200)
        rbt_nodes = []
        lmk_nodes = []
        class2lmk_nodes = {}
        truth = {}
        factors = []

        last_odom_pose = SE3Pose(np.eye(4))

        # prior_cov = np.eye(6) * 0.001

        # visualization setting
        max_color_cnts = 1000
        frustum_scale = 0.03
        colors = np.random.random((max_color_cnts, 3))
        frustum_vertices = np.array([
            [0, 0, 0],
            [1, 0.5, 0.5],
            [-1, 0.5, 0.5],
            [-1, -0.5, 0.5],
            [1, -0.5, 0.5]
        ])
        frustum_vertices = np.hstack((frustum_vertices*frustum_scale, np.ones((len(frustum_vertices),1)))).T

        frustum_lines = [[0, 1],
                         [0, 2],
                         [0, 3],
                         [0, 4],
                         [1, 2],
                         [2, 3],
                         [3, 4],
                         [4, 1]]

        frustum_colors = [[1, 0, 0] for i in range(len(frustum_lines))]

        step_list = []
        step_timer = []
        cur_truth = []
        for step, f_id in enumerate(kf_ids):
            # step_nodes = []
            # detected_lmks = []
            for tmp_k, tmp_v in perf_dict.items():
                if isinstance(tmp_v, list):
                    tmp_v.append(.0)

            start = time.time()
            rbt_var = SE3Variable(f"X{f_id}")
            if gt_poses is not None:
                cur_truth.append(gt_poses[step].mat)

            add_lmk = False

            if step == 0:
                f = UnarySE3Factor(var=rbt_var, prior_pose=odom_poses[step], covariance=prior_cov)
                factors.append(f)
                solver.add_prior_factor(f)
                time_s = time.time()
                solver.update_gs_estimate()
                perf_dict["GA"][-1] += time.time() - time_s
                add_lmk = True
            else:
                tmp_odom_tf = last_odom_pose.inverse() * odom_poses[step]
                f = SE3RelativeGaussianLikelihoodFactor(var1=rbt_nodes[-1], var2=rbt_var,
                                                        observation=tmp_odom_tf,
                                                        covariance=odom_cov)
                factors.append(f)
                solver.add_odom_factor(f)
                time_s = time.time()
                solver.update_gs_estimate()
                perf_dict["GA"][-1] += time.time() - time_s

                if np.linalg.norm(tmp_odom_tf.translation) > delta_trans_tol or abs(np.arccos(np.clip((np.trace(tmp_odom_tf.rotation)-1)/2, -1, 1))) > delta_rot_tol:
                    add_lmk = True

            if add_lmk:
                # gtsam.Pose3 object
                cur_rbt_mat = solver.gaussian_solver.get_point_estimate(var=to_Key(rbt_var),
                                                                        vartype=rbt_var.__class__.__name__).matrix()
                cur_rbt_mat_inv = SE3Pose(cur_rbt_mat).inverse().mat
                # insert measurements of object segments
                obj_seg = read_obj_seg_frame(obj_det_dir, f_id, img_prefix, admitted_class)
                added_objects = []

                perf_dict["num_dets"][-1] = len(obj_seg.masks)

                time_s = time.time()
                if len(obj_seg.masks) > 0:
                    obj_box_masks = np.zeros((len(obj_seg.masks), obj_seg.masks[0].shape[0], obj_seg.masks[0].shape[1]),
                                             dtype=bool)
                    obj_box_masks, obj_yx_mins, obj_yx_maxs, obj_seg_pixels = objmasks2geominfo(obj_seg.masks, obj_box_masks)
                else:
                    obj_box_masks, obj_yx_mins, obj_yx_maxs, obj_seg_pixels = [], [], [], []

                # removing small occluded objects and low-score detections

                if len(obj_seg.masks) > 0 and ablation_setting["preprocess"]:
                    occluded_ids = objmasks2occludedIDs(obj_seg.masks, obj_box_masks, min_relative_ratio, max_occluded_ratio)
                else:
                    occluded_ids = []

                if len(lmk_nodes) > 0:
                    lmks_points = np.array(solver.vars2points(lmk_nodes))
                    lmks_pixels = points2pixels(cur_rbt_mat_inv,lmks_points,cam_intrinsic).T
                else:
                    lmks_pixels = []
                perf_dict["pre_process"][-1] += time.time() - time_s

                # cache joint marginal covariance matrices from gtsam to avoid repeated computation
                lmk2jointCov = {}
                for obj_i, obj_class in enumerate(obj_seg.obj_cls):
                    obj_mask = obj_seg.masks[obj_i]
                    obj_score = obj_seg.scores[obj_i]
                    if obj_score>obj_lo_score and obj_i not in occluded_ids and obj_class in admitted_class:
                        h, w = obj_mask.shape
                        obj_pixels = obj_seg_pixels[obj_i]
                        y_center, x_center = np.mean(obj_pixels, axis=0)
                        y_min, x_min = obj_yx_mins[obj_i]
                        y_max, x_max = obj_yx_maxs[obj_i]
                        box_mask = obj_box_masks[obj_i]
                        true_y_std, true_x_std = np.std(obj_pixels, axis=0) #*2
                        if ablation_setting["preprocess"]:
                            relative_x,  relative_y = np.clip(abs(x_center-cx)/(img_w/2), 0, 1), np.clip(abs(y_center-cy)/(img_h/2), 0, 1)
                            x_std = (1.0-relative_x) * true_x_std + relative_x * np.max([true_y_std, true_x_std])
                            y_std = (1.0-relative_y) * true_y_std + relative_y * np.max([true_y_std, true_x_std])
                        else:
                            x_std = true_x_std
                            y_std = true_y_std

                        x_std = max(std_min_cap, x_std)
                        y_std = max(std_min_cap, y_std)
                        # y_std = max(np.std(obj_pixels, axis=0)) #*2
                        # x_std = y_std
                        # y_std = 0.5 * (y_max - y_min)
                        # x_std = 0.5 * (x_max - x_min)

                        # semantic data association
                        meas_xy, meas_cov = [x_center, y_center], np.diag([x_std**2, y_std**2])

                        if len(lmks_pixels) > 0:
                            euc_distances = np.linalg.norm(lmks_pixels - meas_xy, axis=1)
                            close_lmks_idx = np.where(euc_distances < pixel_range)[0]
                            close_lmks = set([lmk_nodes[tmp_i] for tmp_i in close_lmks_idx])
                        else:
                            close_lmks = set()
                        if obj_class in class2lmk_nodes:
                            # performing MLE data association to landmarks with the same class
                            same_class_lmks = close_lmks.intersection(class2lmk_nodes[obj_class])
                            diff_class_lmks = close_lmks-same_class_lmks
                        else:
                            same_class_lmks = []
                            diff_class_lmks = close_lmks
                        same_class_ng_lmks = list(set(same_class_lmks) - set(solver.unimlmk))
                        same_class_gs_lmks = list(set(same_class_lmks) - set(same_class_ng_lmks))
                        same_class_lmks = list(same_class_lmks)
                        diff_class_lmks = list(diff_class_lmks)

                        associated_lmk = None
                        # semantic +ML data association

                        time_s = time.time()
                        if len(same_class_lmks) > 0:
                            mahalanobis_distances = MahalanobisDistances(same_class_lmks, solver, rbt_var, meas_xy, meas_cov, cur_rbt_mat, cur_rbt_mat_inv, cam_intrinsic, cam_model, lmk2jointCov)
                            cur_min_dist = np.min(mahalanobis_distances)
                            # print(f"current min dist: {cur_min_dist} lmk class {obj_class}")
                            if cur_min_dist < min_mhl_dist:
                                associated_lmk = same_class_lmks[np.argmin(mahalanobis_distances)]
                        perf_dict["da"][-1] += time.time() - time_s

                        time_s = time.time()
                        if associated_lmk is None and len(diff_class_lmks)>0 and obj_class in class2lmk_nodes and ablation_setting["preprocess"]:
                            # reject object detection with wrong semantic labels
                            diff_class_distances = MahalanobisDistances(diff_class_lmks, solver, rbt_var, meas_xy, meas_cov, cur_rbt_mat, cur_rbt_mat_inv, cam_intrinsic, cam_model, lmk2jointCov)
                            if np.min(diff_class_distances) < min_mhl_dist:
                                # give up this object detection since it is too close to an object of different classes
                                # print(f"rejecting a detection of {obj_class}")
                                continue
                        perf_dict["pre_process"][-1] += time.time() - time_s

                        time_s = time.time()
                        if associated_lmk is None and len(same_class_ng_lmks) > 0 and ablation_setting["sampleDA"]:
                            # using samples to verify the new landmark
                            # data association using landmark samples
                            sample_pct_in_box = []
                            for tmp_i, lmk_var in enumerate(same_class_ng_lmks):
                                pixels_in_cam = points2pixels(cur_rbt_mat_inv, solver.lmk_samples[lmk_var], cam_intrinsic)
                                pixels_in_box = (pixels_in_cam[0, :] >= x_min) & (pixels_in_cam[0, :] <= x_max) & (
                                        pixels_in_cam[1, :] >= y_min) & (pixels_in_cam[1, :] <= y_max)
                                sample_pct_in_box.append(1.0*sum(pixels_in_box)/len(pixels_in_box))
                            # print(f"current percents of box points: {sample_pct_in_box}")
                            if len(sample_pct_in_box) > 0 and np.max(sample_pct_in_box) > box_pts_min_percent:
                                associated_lmk = same_class_ng_lmks[np.argmax(sample_pct_in_box)]

                        if associated_lmk is None and len(same_class_gs_lmks) > 0 and ablation_setting["sampleDA"]:
                            sample_pct_in_box = []
                            iou_list = []
                            for lmk_var in same_class_gs_lmks:
                                center = solver.gaussian_solver.get_point_estimate(var=to_Key(lmk_var), vartype=lmk_var.__class__.__name__)
                                cov = solver.gaussian_solver.get_single_covariance(lmk_var.key)
                                xyzpts, az, el = xyzcov2ellipsoidpts(center, cov, n_std=1.0, resolution=50)

                                pixels_in_cam = points2pixels(cur_rbt_mat_inv, xyzpts, cam_intrinsic)

                                pixels_in_box = (pixels_in_cam[0, :] >= x_min) & (
                                            pixels_in_cam[0, :] <= x_max) & (
                                                        pixels_in_cam[1, :] >= y_min) & (
                                                            pixels_in_cam[1, :] <= y_max)
                                sample_pct_in_box.append(1.0 * sum(pixels_in_box) / len(pixels_in_box))
                                if sum(pixels_in_box) > 0:
                                    tmp_x_min, tmp_y_min = np.min(pixels_in_cam[:, np.arange(len(pixels_in_box))[pixels_in_box]], axis=1).astype(int)
                                    tmp_x_max, tmp_y_max = np.max(pixels_in_cam[:, np.arange(len(pixels_in_box))[pixels_in_box]], axis=1).astype(int)
                                    distribution_mask = np.zeros_like(box_mask, dtype=bool)
                                    distribution_mask[tmp_y_min:tmp_y_max+1, tmp_x_min:tmp_x_max+1] = True
                                    iou_list.append(1.0*np.sum(distribution_mask&box_mask)/np.sum(distribution_mask|box_mask))
                                else:
                                    iou_list.append(0)
                            if len(sample_pct_in_box)>0:
                                if np.max(sample_pct_in_box) > box_pts_min_percent:
                                    associated_lmk = same_class_gs_lmks[np.argmax(sample_pct_in_box)]
                                elif np.max(iou_list) > box_pts_min_percent:
                                    associated_lmk = same_class_gs_lmks[np.argmax(iou_list)]
                            # print(f"current Gaussian lmks IOU: {iou_list} percents of box points: {sample_pct_in_box}")

                        perf_dict["da"][-1] += time.time() - time_s

                        if associated_lmk is None:
                            new_lmk = R3Variable(name=f"L{len(lmk_nodes)}", variable_type=VariableType.Landmark)
                            lmk_nodes.append(new_lmk)
                            if obj_class not in class2lmk_nodes:
                                class2lmk_nodes[obj_class] = [new_lmk]
                            else:
                                class2lmk_nodes[obj_class].append(new_lmk)
                            cur_lmk = new_lmk
                            # print(f"creating new landmark: {new_lmk} with class {obj_class}")
                        else:
                            # print(f"associated landmark: {associated_lmk.name} with class {obj_class}")
                            cur_lmk = associated_lmk
                        f = CameraProjectionFactor(var1=rbt_var, var2=cur_lmk,
                                                   observation=np.array([x_center, y_center]),
                                                   covariance=np.diag(np.array([x_std, y_std]) ** 2),
                                                   cam_params=np.array(cam_params),
                                                   min_depth=min_depth,
                                                   max_depth=max_depth)
                        reinit_time_list = []
                        solver.add_lmk_meas_factor(f, reinit_time_list, ablation_setting["reinit"])
                        if len(reinit_time_list) > 0:
                            perf_dict['reinit'][-1] += reinit_time_list[0]
                        time_s = time.time()
                        solver.update_gs_estimate()
                        perf_dict['GA'][-1] += time.time() - time_s
                        time_s = time.time()
                        solver.update_lmk_samples(lmk_var=cur_lmk, path_num=rbt_path_num,
                                                  sample_per_path=sample_per_path, downsample=1000,
                                                  eig_threshold=eig_threshold)
                        perf_dict['sampling'][-1] += time.time() - time_s
                        factors.append(f)
                        added_objects.append(obj_class)
                # print(f"addedlandmarks {added_objects} at time step {step} (KF{f_id})")
            rbt_nodes.append(rbt_var)
            last_odom_pose = odom_poses[step]

            perf_dict['step_time'][-1] += time.time() - start
            perf_dict['gapslam'][-1] = perf_dict['step_time'][-1] - perf_dict['pre_process'][-1] - perf_dict['da'][-1]
            # lmk_sample_dict = solver.lmk_samples

            step_list.append(step)
            step_file_prefix = f"{run_dir}/step{step}"
            end = time.time()
            step_timer.append(end - start)
            # print(f"step {step}/{len(kf_ids)} time: {step_timer[-1]} sec, "
            #       f"total time: {sum(step_timer)}")

            if step == len(kf_ids) - 1:
                gs_mean, gs_cov = solver.get_gs_marginals()
                cam_pose_mats = []

                for v_i, v in enumerate(solver.gs_vars):
                    if isinstance(v, SE3Variable):
                        cam_pose_mats.append(SE3Pose.by_trans_rotvec(gs_mean[v_i]).mat)

                file = open(f"{run_dir}/step_timing", "w+")
                file.write(" ".join(str(t) for t in step_timer))
                file.close()
                file = open(f"{run_dir}/step_list", "w+")
                file.write(" ".join(str(s) for s in step_list))
                file.close()

                fig2, ax2 = plt.subplots()
                ax2.plot(step_list, step_timer, 'go-', label='Total')
                ax2.set_ylabel(f"Time (sec)")
                ax2.set_xlabel(f"Key poses")
                ax2.legend()
                fig2.savefig(f"{run_dir}/step_timing.png", bbox_inches="tight")
                plt.close(fig2)

                traj_est = PosePath3D(poses_se3=cam_pose_mats)
                traj_odom = PosePath3D(poses_se3=odom_mats[:step + 1])
                traj_ref = PosePath3D(poses_se3=cur_truth)
                traj_est_aligned_scaled = deepcopy(traj_est)
                est_res = traj_est_aligned_scaled.align(traj_ref, correct_only_scale=True)
                traj_odom_aligned_scaled = deepcopy(traj_odom)
                odom_res = traj_odom_aligned_scaled.align(traj_ref, correct_only_scale=True)

                f_size = 24
                plt.rc('xtick', labelsize=f_size)
                plt.rc('ytick', labelsize=f_size)

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.plot(traj_ref.positions_xyz[:, 0], traj_ref.positions_xyz[:, -1], linestyle='-', color='gray',
                        label="Pseudo ground truth")
                ax.plot(traj_est_aligned_scaled.positions_xyz[:, 0], traj_est_aligned_scaled.positions_xyz[:, -1],
                        linestyle='--', color='blue', label="GAPSLAM")
                ax.plot(traj_odom_aligned_scaled.positions_xyz[:, 0], traj_odom_aligned_scaled.positions_xyz[:, -1],
                        linestyle='-.', color='red', label="ORBSLAM3")
                ax.set_aspect('equal', 'box')
                ax.set_xlabel("x (m)", fontsize=f_size)
                ax.set_ylabel("z (m)", fontsize=f_size)
                ax.legend(bbox_to_anchor=(0.5, 0.65), loc='center', fontsize=f_size - 4)
                fig.tight_layout()
                fig.savefig(f"{run_dir}/final_traj.png", dpi=300)
                plt.close(fig)

                pose_relation = metrics.PoseRelation.translation_part
                ape_metric = metrics.APE(pose_relation)
                ape_metric.process_data((traj_ref, traj_est_aligned_scaled))
                ape_stats = ape_metric.get_all_statistics()

                ape_metric2 = metrics.APE(pose_relation)
                ape_metric2.process_data((traj_ref, traj_odom_aligned_scaled))
                ape_stats2 = ape_metric2.get_all_statistics()
                # pprint.pprint(ape_stats2)
                #
                #
                perf_dict['steps'] = len(kf_ids)
                perf_dict['num_lmks'] = len(lmk_nodes)
                perf_dict['odom_rmse'] = ape_stats2["rmse"]
                perf_dict['gap_rmse'] = ape_stats["rmse"]
                # exclude the overhead in the first time step
                perf_dict['mean_step_time'] = np.mean(perf_dict['step_time'][1:])
                perf_dict['mean_preprocess'] = np.mean(perf_dict['pre_process'][1:])
                perf_dict['mean_da'] = np.mean(perf_dict['da'][1:])
                perf_dict['mean_gap'] = np.mean(perf_dict['gapslam'][1:])
                perf_dict['mean_GA'] = np.mean(perf_dict['GA'][1:])
                perf_dict['mean_sampling'] = np.mean(perf_dict['sampling'][1:])
                perf_dict['mean_det'] = np.mean(perf_dict['num_dets'])
                perf_dict['mean_reinit'] = np.mean(perf_dict['reinit'][1:])
                with open(f'{run_dir}/perf{step}.json', 'w', encoding='utf-8') as f:
                    json.dump(perf_dict, f, ensure_ascii=False, indent=4)


