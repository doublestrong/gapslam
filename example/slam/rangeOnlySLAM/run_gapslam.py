import sys

import os
import random
import time

import numpy as np
from matplotlib import pyplot as plt

from factors.Factors import PriorFactor, OdomFactor
from slam.RunBatch import graph_file_parser, group_nodes_factors_incrementally
from adaptive_inference.GAPSLAM import GAPSLAM
from slam.Variables import VariableType
from utils.Visualization import plot_2d_samples, xycov2ellipse
import logging

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    cases = ["Plaza1","Plaza2"]
    max_steps = [28, 20]

    cases = ["Plaza1"]
    max_steps = [28]

    # rd_seeds = np.arange(50)
    rd_seeds = np.arange(1)

    only_save_last_step = True
    no_samples = True

    posterior_sample_num = 2000


    markers = ['o','v','^','<','>','1','2','3','4','8','s','p','*','h','H','+','x','D','d']
    colors = []
    for i in range(len(markers)):
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))

    var2color = {}
    traj_plot = True
    plot_args = {'truth_label_offset': (3, -3), 'show_plot': False}
    incremental_step = 1

    case_prefix = "batch_gap"

    rbt_path_for_lmk_sampling = 200
    lmk_sampling_per_path = 100

    for rd_seed in rd_seeds:
        random.seed(rd_seed)
        np.random.seed(rd_seed)
        for case_i, case in enumerate(cases):
            case_dir = f"RangeOnlyDataset/{case}EFG"
            data_file = 'factor_graph.fg'
            data_format = 'fg'

            data_dir = os.path.join(case_dir, data_file)
            nodes, truth, factors = graph_file_parser(data_file=data_dir, data_format=data_format,
                                                      prior_cov_scale=1)
            color_idx=0
            for v in nodes:
                if v.type == VariableType.Landmark:
                    var2color[v] = colors[color_idx]
                    color_idx+=1
                else:
                    var2color[v] = "gray"
            nodes_factors_by_step = group_nodes_factors_incrementally(
                nodes=nodes, factors=factors, incremental_step=incremental_step)
            # nodes_factors_by_step = nodes_factors_by_step[:max_steps[case_i]+1]

            output_dir = case_dir
            run_count = 1
            while os.path.exists(f"{output_dir}/{case_prefix}{run_count}"):
                run_count += 1
            os.mkdir(f"{output_dir}/{case_prefix}{run_count}")
            run_dir = f"{output_dir}/{case_prefix}{run_count}"
            print("create run dir: " + run_dir)

            num_batches = len(nodes_factors_by_step)
            observed_nodes = []
            step_timer = []
            step_list = []

            posterior_sampling_timer = []
            fitting_timer = []

            mixture_factor2weights = {}

            show_plot = True
            if "show_plot" in plot_args and not plot_args["show_plot"]:
                show_plot = False

            solver = GAPSLAM(bootstrap_init_class={'SE2RelativeGaussianLikelihoodFactor','UnarySE2ApproximateGaussianPriorFactor'},
                             bw_method=0.01,
                             prior_cov_scale=50,
                             lmk_sample_num=200,
                             rd_seed=rd_seed)

            for i in range(num_batches):
                detailed_timer = {"GA": .0, "LmkReinit": .0, "AddMeas": .0, "LmkSamples": .0, "TotalUpdate":.0, "LmkPosterior": .0, "NGLMK": 0}
                start = time.time()
                step_nodes, step_factors = nodes_factors_by_step[i]
                prior_f = []
                odom_f = []
                for factor in step_factors:
                    if isinstance(factor, PriorFactor):
                        prior_f.append(factor)
                    elif isinstance(factor, OdomFactor):
                        odom_f.append(factor)
                for f in prior_f:
                    solver.add_prior_factor(f)
                    step_factors.remove(f)
                    time_s = time.time()
                    solver.update_gs_estimate()
                    time_e = time.time()
                    detailed_timer['GA'] += time_e - time_s

                for f in odom_f:
                    solver.add_odom_factor(f)
                    step_factors.remove(f)
                    time_s = time.time()
                    solver.update_gs_estimate()
                    time_e = time.time()
                    detailed_timer['GA'] += time_e - time_s

                for f in step_factors:
                    reinit_time_list = []
                    time_s = time.time()
                    solver.add_lmk_meas_factor(f, reinit_time_list)
                    time_e = time.time()
                    detailed_timer['AddMeas'] += time_e - time_s
                    if len(reinit_time_list) > 0:
                        detailed_timer['LmkReinit'] += reinit_time_list[0]

                    time_s = time.time()
                    solver.update_gs_estimate()
                    time_e = time.time()
                    detailed_timer['GA'] += time_e - time_s

                    time_s = time.time()
                    solver.update_lmk_samples(lmk_var=f.var2, path_num=1, sample_per_path=200,downsample=1000,eig_threshold=3.0)
                    time_e = time.time()
                    detailed_timer['LmkSamples'] += time_e - time_s

                # key_sample_dict=solver.get_ng_lmk_samples(path_num=500, sample_num=500)
                # solver.get_ng_lmk_samples(path_num=50, sample_per_path=100,downsample=1000,eig_threshold=5)
                observed_nodes += step_nodes
                step_list.append(i)
                step_file_prefix = f"{run_dir}/step{i}"
                end = time.time()
                detailed_timer['TotalUpdate'] = end - start
                step_timer.append(detailed_timer['TotalUpdate'])
                print(f"step {i}/{num_batches} time: {step_timer[-1]} sec, "
                      f"total time: {sum(step_timer)}")

                # ng_vars = [v for v in solver.nglmk]
                lmk2color = {v: colors[i] for i, v in enumerate(solver.nglmk)}

                # key_sample_dict = {}
                rbt_vars = [v for v in solver.gs_vars if v not in solver.nglmk]

                detailed_timer['NGLMK'] = len(solver.nglmk) - len(solver.unimlmk)

                # print("-- detailed timing: ")
                # print(detailed_timer)
                # with open(f'{run_dir}/step{i}_time.json', 'w', encoding='utf-8') as f:
                #     json.dump(detailed_timer, f, ensure_ascii=False, indent=4)

                save_fig = False
                if only_save_last_step and i == num_batches-1:
                    save_fig = True

                if save_fig:
                    file = open(f"{step_file_prefix}_ordering", "w+")
                    file.write(" ".join([v.name for v in solver.gs_vars]))
                    file.close()

                    file = open(f"{run_dir}/step_timing", "w+")
                    file.write(" ".join(str(t) for t in step_timer))
                    file.close()
                    file = open(f"{run_dir}/step_list", "w+")
                    file.write(" ".join(str(s) for s in step_list))
                    file.close()

                    fig2, ax2 = plt.subplots()
                    ax2.plot(np.array(step_list) * incremental_step + 1, step_timer, 'go-', label='Total')
                    ax2.set_ylabel(f"Time (sec)")
                    ax2.set_xlabel(f"Key poses")
                    ax2.legend()
                    fig2.savefig(f"{run_dir}/step_timing.png", bbox_inches="tight")
                    if show_plot: plt.show()
                    plt.close(fig2)

                    gs_mean, gs_cov = solver.get_gs_marginals()

                    file = open(f"{step_file_prefix}_mean", "w+")
                    file.write("\n".join([" ".join([str(ele) for ele in arr]) for arr in gs_mean]))
                    file.close()

                    file = open(f"{step_file_prefix}_cov", "w+")
                    file.write("\n".join([" ".join([str(ele) for ele in arr.flatten()]) for arr in gs_cov]))
                    file.close()

                    fig, ax = plt.subplots()
                    key_sample_dict = {tmp_v: np.array([gs_mean[tmp_i]]) for tmp_i, tmp_v in enumerate(solver.gs_vars)}
                    if traj_plot:
                        plot_2d_samples(ax=ax,
                                        samples_mapping=key_sample_dict,
                                        equal_axis=True,
                                        truth={variable: pose for variable, pose in
                                               truth.items() if variable in observed_nodes},
                                        truth_factors={factor for factor in solver.all_factors if
                                                       set(factor.vars).issubset(observed_nodes)},
                                        title=f'Time Step {i}',
                                        colors={v: c for v, c in var2color.items() if v in observed_nodes},
                                        plot_all_meas=False,
                                        plot_meas_give_pose=[var for var in step_nodes if var.type == VariableType.Pose],
                                        rbt_traj_no_samples=True,
                                        truth_R2=True,
                                        truth_SE2=False,
                                        truth_odometry_color='k',
                                        truth_landmark_markersize=10,
                                        truth_landmark_marker='x',
                                        marker_size=10,
                                        **plot_args)
                    else:
                        plot_2d_samples(ax=ax,
                                        samples_mapping=key_sample_dict,
                                        equal_axis=True,
                                        truth={variable: pose for variable, pose in
                                               truth.items() if variable in observed_nodes},
                                        truth_factors={factor for factor in solver.all_factors if
                                                       set(factor.vars).issubset(observed_nodes)},
                                        title=f'Time Step {i}',
                                        colors={v: c for v, c in var2color.items() if v in observed_nodes},
                                        plot_all_meas=True,
                                        # plot_meas_give_pose=[var for var in step_nodes if var.type == VariableType.Pose],
                                        rbt_traj_no_samples=False,
                                        truth_R2=True,
                                        truth_SE2=False,
                                        truth_odometry_color='k',
                                        truth_landmark_markersize=10,
                                        truth_landmark_marker='x',
                                        marker_size=10,
                                        **plot_args)
                    for v_i, v in enumerate(solver.gs_vars):
                        if v.type == VariableType.Landmark:
                            xycov2ellipse(gs_mean[v_i], gs_cov[v_i],
                                          ax, n_std=2.0,
                                          edgecolor="black")
                        # else:
                        #     # if traj_plot:
                        #     ax.scatter(gs_mean[v_i][0], gs_mean[v_i][1], marker='o', c="black", s=5)  # initial ellipse
                        #     se2xy = SE2cov2ellipsoidpts(gs_mean[v_i], gs_cov[v_i], n_std=2.0, resolution=200)
                        #     ax.plot(se2xy[:,0],se2xy[:,1],"black")
                    fig.savefig(f"{step_file_prefix}.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)
