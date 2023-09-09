import os
from sampler.NestedSampling import dynesty_run_batch

if __name__ == '__main__':
    run_file_dir = os.path.dirname(__file__)
    for _ in range(1):
        cases = ["Plaza1"]
        for case in cases:
            case_dir = os.path.join(run_file_dir,f"RangeOnlyDataset/{case}EFG")
            data_file = 'factor_graph_earlySteps.fg'
            data_format = 'fg'
            dynesty_run_batch(1000, case_dir, data_file, data_format,
                              incremental_step=1, parallel_config={'cpu_frac': 0.8, 'queue_size': 64}, prior_cov_scale=0.1,
                              plot_args={'truth_label_offset': (3, -3), 'show_plot': False})
