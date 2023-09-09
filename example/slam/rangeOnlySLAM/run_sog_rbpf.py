import json
import logging, sys

import os
import random
import time

import numpy as np
from matplotlib import pyplot as plt

from factors.Factors import PriorFactor, OdomFactor
from sampler.RBPFSOG import RBPFSOG
from slam.RunBatch import graph_file_parser, group_nodes_factors_incrementally
from slam.Variables import VariableType
from utils.Visualization import plot_2d_samples

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    cases = ["Plaza1"]
    # max_steps = [28]

    min_ess_ratio = 0.8

    case_prefix = "batch_sog"

    only_save_last_step = True
    traj_plot = True
    plot_args = {'truth_label_offset': (3, -3), 'show_plot': False}
    incremental_step = 1
    # rd_seeds = np.arange(50)
    rd_seeds = np.arange(1)
    # rd_seeds = [0]
    for rd_seed in rd_seeds:
        random.seed(rd_seed)
        np.random.seed(rd_seed)

        # posterior_sample_num = 2000
        posterior_sample_num = 2000

        markers = ['o','v','^','<','>','1','2','3','4','8','s','p','*','h','H','+','x','D','d']
        colors = []
        for i in range(len(markers)):
            colors.append('#%06X' % random.randint(0, 0xFFFFFF))

        var2color = {}
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

            solver = RBPFSOG(50, 2500, max_drop_mode_weight=0.8)
            output_dir = case_dir

            run_count = 1
            while os.path.exists(f"{output_dir}/{case_prefix}{run_count}"):
                run_count += 1
            os.mkdir(f"{output_dir}/{case_prefix}{run_count}")
            run_dir = f"{output_dir}/{case_prefix}{run_count}"
            print("create run dir: " + run_dir)

            for i in range(num_batches):
                print(f"Time step {i}")
                detailed_timer = {"TotalUpdate": .0, "LmkPosterior": .0, "AvgNumMode": 0}
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
                for f in odom_f:
                    solver.add_odom_factor(f)
                    step_factors.remove(f)
                for f in step_factors:
                    solver.add_lmk_meas_factor(f)

                ess = 1.0 / np.sum(solver.weights ** 2)
                print(f"current ESS ratio {ess / len(solver.weights)}")
                if ess / len(solver.weights) < min_ess_ratio:
                    print("-- resampling...")
                    solver.resample()

                # key_sample_dict=solver.get_ng_lmk_samples(path_num=500, sample_num=500)
                # solver.get_ng_lmk_samples(path_num=50, sample_per_path=100,downsample=1000,eig_threshold=5)
                observed_nodes += step_nodes
                step_list.append(i)
                end = time.time()
                step_timer.append(end - start)

                if only_save_last_step and i == num_batches - 1:
                    step_file_prefix = f"{run_dir}/step{i}"
                    print(f"step {i}/{num_batches} time: {step_timer[-1]} sec, "
                          f"total time: {sum(step_timer)}")
                    detailed_timer["TotalUpdate"] = step_timer[-1]


                    sampling_start = time.time()

                    sample_per_path, lmk_sample_dict, rbt_sample_dict = solver.sample_lmk(posterior_sample_num)

                    lmk_posterior_time = time.time() - sampling_start
                    if len(solver.lmk_vars) > 0:
                        detailed_timer["LmkPosterior"] = lmk_posterior_time / len(solver.lmk_vars)
                        tmp_time = detailed_timer["LmkPosterior"]
                        print(f"-- posterior sampling time per landmark {tmp_time}")

                    agv_lmk_modes = solver.getAvgModeNum()
                    detailed_timer["AvgNumMode"] = np.mean(list(agv_lmk_modes.values()))
                    detailed_timer = {**detailed_timer, **agv_lmk_modes}

                    print(f"-- detailed results")
                    print(detailed_timer)
                    with open(f'{run_dir}/step{i}_time.json', 'w', encoding='utf-8') as f:
                        json.dump(detailed_timer, f, ensure_ascii=False, indent=4)
                    # ng_vars = solver.lmk_vars
                    lmk2color = {v: colors[i] for i, v in enumerate(solver.lmk_vars)}

                    key_sample_dict = {**lmk_sample_dict, **rbt_sample_dict}
                    samples = [key_sample_dict[k] for k in solver.vars]
                    samples = np.hstack(samples)
                    np.savetxt(fname=f"{step_file_prefix}.sample", X=samples)

                    file = open(f"{step_file_prefix}_ordering", "w+")
                    file.write(" ".join([k.name for k in solver.vars]))
                    file.close()

                    fig, ax = plt.subplots()
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
                                        colors={v: c for v, c in var2color.items() if v in observed_nodes},
                                        title=f'Time Step {i}',
                                        truth_R2=True,
                                        truth_SE2=False,
                                        truth_odometry_color='k',
                                        truth_landmark_markersize=10,
                                        truth_landmark_marker='x',
                                        marker_size=10,
                                        **plot_args)
                    # for v_i, v in enumerate(solver.gs_vars):
                    #     if v.type == VariableType.Landmark:
                    #         xycov2ellipse(gs_mean[v_i], gs_cov[v_i],
                    #                       ax, n_std=2.0,
                    #                       edgecolor="black")
                    #                       # edgecolor = var2color[v])
                    fig.savefig(f"{step_file_prefix}.png", dpi=300, bbox_inches="tight")
                    plt.close(fig)

                    file = open(f"{run_dir}/step_timing", "w+")
                    file.write(" ".join(str(t) for t in step_timer))
                    file.close()
                    file = open(f"{run_dir}/step_list", "w+")
                    file.write(" ".join(str(s) for s in step_list))
                    file.close()

                    fig2, ax2=plt.subplots()
                    ax2.plot(np.array(step_list) * incremental_step + 1, step_timer, 'go-', label='Total')
                    ax2.set_ylabel(f"Time (sec)")
                    ax2.set_xlabel(f"Key poses")
                    ax2.legend()
                    fig2.savefig(f"{run_dir}/step_timing.png", bbox_inches="tight")
                    if show_plot: plt.show()
                    plt.close(fig2)