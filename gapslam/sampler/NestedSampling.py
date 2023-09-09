import json
import multiprocessing as mp
import os
import random
import time
from typing import List

import numpy as np
from dynesty import NestedSampler, DynamicNestedSampler, utils as dyfunc
from matplotlib import pyplot as plt

from sampler.sampler_utils import JointLikelihoodForNestedSampler, JointFactorForNestedSampler
from factors.Factors import Factor, BinaryFactorMixture, KWayFactor
from slam.RunBatch import graph_file_parser, group_nodes_factors_incrementally
from slam.Variables import Variable
from utils.Functions import NumpyEncoder
from utils.Visualization import plot_2d_samples


class GlobalNestedSampler(object):
    def __init__(self, nodes: List[Variable], factors: List[Factor], xlim:list=None,ylim:list=None, *args, **kwargs):
        self._dim = sum([var.dim for var in nodes])
        if xlim is not None and ylim is not None:
        # naive nested sampling propose prior from uniform distribution
            self._joint_factor = JointLikelihoodForNestedSampler(factors = factors,
                                                                 variable_pattern = nodes,
                                                                 x_lim = xlim,
                                                                 y_lim = ylim)
        else:
        # propose prior from an acyclic factor graph
            self._joint_factor = JointFactorForNestedSampler(factors=factors, variable_pattern=nodes, *args, **kwargs)

    def sample(self, live_points: int, sampling_method: str = "nested", downsampling = False,
               maxiter:int = None, maxcall:int = None, dlogz = .05, dns_params = None,ns_params = None, use_grad_u:bool = False,
               adapt_live_pt=False,
               res_summary={},
               **kwargs
               ) -> np.ndarray:
        """
        downsampling: draw live_points samples from converged results and return them
        sampling_method: any str rather than nested leads to dynamic nested sampling
        """
        joint_factor = self._joint_factor
        dim = self._dim
        print(f" Dim of problem: {dim}")
        ns_start = time.time()
        if joint_factor.ifDirectSampling:
            # direct sampling
            print("  using ancestral sampling")
            local_samples = joint_factor.sample(live_points)
        else:
            # nested sampling
            # caution! nlive is not the number of samples but can represent the magnitude of sample number.
            if adapt_live_pt:
                seed_num = dim * 50 # (dim + 1) *3 //4 #50
                dlogz *= dim / 105
            else:
                seed_num = live_points
            print(f" Number of seeds: {seed_num}")

            # if random_method is None:
            #     if dim <= 10:
            #         random_method = "unif"
            #     else:
            #         random_method = "rwalk"
            # if dim * (dim + 1) > (seed_num + 500):
            #     seed_num = dim * (dim + 1)
            # else:
            #     seed_num = seed_num + 500

            if maxiter is None:
                maxiter = seed_num * 100
            if maxcall is None:
                maxcall = seed_num * 10000

            print(f" maxcall: {maxcall}")
            print(f" maxiter: {maxiter}")


            if use_grad_u and hasattr(joint_factor, 'grad_u_loglike'):
                grad_u = joint_factor.grad_u_loglike
            else:
                grad_u = None

            # circ_idx = np.arange(len(joint_factor.circular_dim_list))[joint_factor.circular_dim_list]
            # if len(circ_idx) == 0:
            #     periodic = None
            # else:
            #     periodic = circ_idx

            print("sampler kwargs:")
            print(kwargs)
            if sampling_method == "nested":
                print("  using nested sampling")
                sampler = NestedSampler(loglikelihood=joint_factor.loglike,
                                        prior_transform=joint_factor.ptform,
                                        # periodic=periodic,
                                        gradient=grad_u,
                                        ndim=dim, nlive=seed_num,
                                        **kwargs
                                        )
                if ns_params is None:
                    sampler.run_nested(maxiter=maxiter,
                                       maxcall=maxcall,
                                       dlogz=dlogz, add_live=False)
                    #, n_effective=2000
                else:
                    print(ns_params)
                    sampler.run_nested(dlogz=dlogz, add_live=True, **ns_params)
            else:
                print("  using dynamic nested sampling")
                sampler = DynamicNestedSampler(
                    loglikelihood=joint_factor.loglike,
                    prior_transform=joint_factor.ptform,
                    # periodic=periodic,
                    gradient=grad_u,
                    ndim=dim,
                    nlive=seed_num,
                    **kwargs)
                if dns_params is None:
                    sampler.run_nested(dlogz_init=dlogz, nlive_init=seed_num,
                                       maxbatch=0,
                                       maxcall_init=maxcall,
                                       maxiter_init=maxiter,
                                       wt_kwargs={'pfrac': 1.0},
                                       use_stop=False)
                else:
                    print(dns_params)
                    sampler.run_nested(**dns_params)
            results = sampler.results
            samples = sampler.results.samples  # samples
            weights = np.exp(results.logwt - results.logz[-1])  # normalized weights
            # mean, cov = dyfunc.mean_and_cov(samples, weights)
            # print("raw weighted sample mean: ", mean)
            # print("raw weighted sample cov: ", cov)
            local_samples = dyfunc.resample_equal(samples, weights)
            #downsampling
            if downsampling:
                if local_samples.shape[0] > live_points:
                    sampled_idx = random.sample(list(range(local_samples.shape[0])), live_points)
                    local_samples = local_samples[sampled_idx,:]
            # print("re-sample mean: ", np.mean(local_samples, axis=0))
            # save summary to files
            items = ['nlive', 'niter', 'ncall', 'eff', 'logz','logzerr']
            for item in items:
                if item in results:
                    if isinstance(results[item], (np.ndarray, list)):
                        res_summary[item] = int(sum(results[item])) if np.issubdtype(results[item][-1], np.integer) \
                            else float(results[item][-1])
                    else:
                        res_summary[item] = results[item]

        # print("re-sample cov: ", cov)
        ns_end = time.time()
        print("Sampling time: " + str(ns_end - ns_start) + " sec")
        return local_samples


def dynesty_run_batch(live_points, case_dir, data_file, data_format,
                      incremental_step=1, selected_steps = None, parallel_config = None, prior_cov_scale=0.1, plot_args=None, dynamic_ns=False,
                      xlim = None,
                      ylim = None,
                      **kwargs):
    data_dir = os.path.join(case_dir, data_file)
    nodes, truth, factors = graph_file_parser(data_file=data_dir, data_format=data_format, prior_cov_scale=prior_cov_scale)

    nodes_factors_by_step = group_nodes_factors_incrementally(
        nodes=nodes, factors=factors, incremental_step=incremental_step)

    run_count = 1
    while os.path.exists(f"{case_dir}/dyn{run_count}"):
        run_count += 1
    os.mkdir(f"{case_dir}/dyn{run_count}")
    run_dir = f"{case_dir}/dyn{run_count}"
    print("create run dir: "+run_dir)
    print("saving config of sampling")
    with open(run_dir+'/config.json', 'w') as fp:
        json.dump(kwargs, fp)

    num_batches = len(nodes_factors_by_step)
    observed_nodes = []
    observed_factors = []
    step_timer = []
    step_list = []

    mixture_factor2weights = {}

    if not dynamic_ns:
        sampling_method = "nested"
    else:
        sampling_method = "dynamic"

    for i in range(num_batches):
        step_nodes, step_factors = nodes_factors_by_step[i]
        observed_nodes += step_nodes
        observed_factors += step_factors
        for factor in step_factors:
            if isinstance(factor, BinaryFactorMixture):
                mixture_factor2weights[factor] = []
        if selected_steps is None or i in selected_steps:
            solver = GlobalNestedSampler(nodes=observed_nodes, factors=observed_factors, xlim=xlim, ylim=ylim)

            res_summary = {}

            step_list.append(i)
            step_file_prefix = f"{run_dir}/step{i}"
            start = time.time()
            if parallel_config is None:
                sample_arr = solver.sample(live_points=live_points,sampling_method=sampling_method, res_summary=res_summary, **kwargs)
            else:
                if 'cpu_frac' in parallel_config:
                    pool = mp.Pool(int(mp.cpu_count()*parallel_config['cpu_frac']))
                else:
                    pool = mp.Pool(int(mp.cpu_count()*.5))
                sample_arr = solver.sample(live_points=live_points, pool=pool, queue_size=parallel_config['queue_size'],
                                           sampling_method=sampling_method, res_summary=res_summary, **kwargs)
            end = time.time()

            with open(f'{step_file_prefix}.summary', 'w+') as smr_fp:
                smr_fp.write(json.dumps(res_summary, cls=NumpyEncoder))

            cur_sample = {}
            cur_dim = 0
            for var in observed_nodes:
                cur_sample[var] = sample_arr[:,cur_dim:cur_dim + var.dim]
                cur_dim += var.dim

            step_timer.append(end - start)
            print(f"step {i}/{num_batches} time: {step_timer[-1]} sec, "
                  f"total time: {sum(step_timer)}")

            file = open(f"{step_file_prefix}_ordering", "w+")
            file.write(" ".join([var.name for var in observed_nodes]))
            file.close()

            X = np.hstack([cur_sample[var] for var in observed_nodes])
            np.savetxt(fname=step_file_prefix+'.sample', X=X)

            plot_2d_samples(samples_mapping=cur_sample,
                            truth={variable: pose for variable, pose in
                                truth.items() if variable in observed_nodes},
                            truth_factors={factor for factor in observed_factors if
                                           set(factor.vars).issubset(observed_nodes)},
                            file_name=f"{step_file_prefix}.png", title=f'Step {i}',
                            **plot_args)

            file = open(f"{run_dir}/step_timing", "w+")
            file.write(" ".join(str(t) for t in step_timer))
            file.close()
            file = open(f"{run_dir}/step_list", "w+")
            file.write(" ".join(str(s) for s in step_list))
            file.close()

            # plt.figure()
            # plt.plot(step_list, step_timer, 'go-')
            # plt.ylabel(f"Time (sec)")
            #
            # plt.savefig(f"{run_dir}/step_timing.png", bbox_inches="tight")
            # plt.show()
            # plt.close()

            if mixture_factor2weights:
                # write updated hypothesis weights
                hypo_file = open(run_dir + f'/step{i}.hypoweights', 'w+')
                plt.figure()
                for factor, weights in mixture_factor2weights.items():
                    hypo_weights = factor.posterior_weights(cur_sample)
                    line = ' '.join([var.name for var in factor.vars]) + ' : ' + ','.join(
                        [str(w) for w in hypo_weights])
                    hypo_file.writelines(line+'\n')
                    weights.append(hypo_weights)
                    for i_w in range(len(hypo_weights)):
                        plt.plot(np.arange(i+1-len(weights), i+1), np.array(weights)[:, i_w],'-o',
                                 label=f"H{i_w}at{factor.observer_var.name}" if not isinstance(factor, KWayFactor) else
                                 f"{factor.observer_var.name} to {factor.observed_vars[i_w].name}")
                hypo_file.close()
                plt.legend()
                plt.xlabel('Step')
                plt.ylabel('Hypothesis weights')
                plt.savefig(run_dir + f'/step{i}_hypoweights.png', dpi=300)
                plt.show()