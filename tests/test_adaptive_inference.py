import unittest
import numpy.testing as test
from adaptive_inference.AdaptiveInference import *
from adaptive_inference.GTSAM_help import *

from adaptive_inference.utils import to_Key
from utils.Visualization import plot_2d_samples


class gtsamWrapperTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vars = [SE2Variable('X0'), SE2Variable('X1'), SE2Variable('X2'), R2Variable('L1', variable_type=VariableType.Landmark)]
        cls.keys = [to_Key(var) for var in cls.vars]
        cls.vartypes = [var.__class__.__name__ for var in cls.vars]

        std = .01
        cls.prior_factor = UnarySE2ApproximateGaussianPriorFactor(var=cls.vars[0],
                                                                  prior_pose=SE2Pose(.0, .0, .0),
                                                                  covariance=np.diag([std]*3))
        cls.odom_factor1 = SE2RelativeGaussianLikelihoodFactor(var1=cls.vars[0],
                                                              var2=cls.vars[1],
                                                              observation=SE2Pose(2.0, .0, np.pi/2),
                                                              covariance=np.diag([std]*3))
        cls.odom_factor2 = SE2RelativeGaussianLikelihoodFactor(var1=cls.vars[1],
                                                              var2=cls.vars[2],
                                                              observation=SE2Pose(2.0, .0, .0),
                                                              covariance=np.diag([std]*3))
        cls.range_factor1 = SE2R2RangeGaussianLikelihoodFactor(var1=cls.vars[0],
                                                              var2=cls.vars[3],
                                                              observation=2.0,
                                                              sigma=.1
                                                              )
        cls.range_factor2 = SE2R2RangeGaussianLikelihoodFactor(var1=cls.vars[1],
                                                              var2=cls.vars[3],
                                                              observation=2.0 * np.sqrt(2),
                                                              sigma=.1)
        cls.range_factor3 = SE2R2RangeGaussianLikelihoodFactor(var1=cls.vars[2],
                                                              var2=cls.vars[3],
                                                              observation=2.0,
                                                              sigma=.1)

        cls.factors = [cls.prior_factor, cls.odom_factor1, cls.odom_factor2, cls.range_factor1, cls.range_factor2, cls.range_factor3]

        cls.arrays = [[.0,.0,.0], [2.0,.0,np.pi/2], [2.0, 2.0, np.pi/2], [.0, 2.0]]

        cls.groundtruth = (gtsam.Pose2(*[.0,.0,.0]), gtsam.Pose2(*[2.0,.0,np.pi/2]), gtsam.Pose2(*[2.0, 2.0, np.pi/2]), gtsam.Point2(*[.0, 2.0]))
        cls.lmk_factor_indices = [3,4,5]
        cls.gtsam_factors = [to_gtsam_factor(f) for f in cls.factors]

    def setupSolver(self):
        solver = AdaptiveInferenceSolver()
        return solver

    def test_update_factor_value(self):
        solver = self.setupSolver()
        show_plot = False
        # add prior
        flag_ambiguous_lmk, flag_uninit_lmk = solver.add_factor(self.prior_factor)
        self.assertFalse(flag_uninit_lmk)
        self.assertFalse(flag_ambiguous_lmk)
        self.assertEqual(solver.gaussian_solver.get_factor_size(), 1)
        init_pose = solver.gaussian_solver.isam.getLinearizationPoint().atPose2(self.keys[0])
        test.assert_almost_equal([init_pose.x(), init_pose.y(), init_pose.theta()], [0,0,0])
        self.assertEqual(set(solver.gaussian_vars), {self.vars[0]})
        est_pose = solver.gaussian_solver.get_point_estimate(self.keys[0], self.vars[0].__class__.__name__)
        test.assert_almost_equal([est_pose.x(), est_pose.y(), est_pose.theta()], [0, 0, 0])
        solver.update_lmk_belief()
        samples, vars = solver.posterior_samples()
        self.assertEqual(set(vars), {self.vars[0]})
        cov = solver.gaussian_solver.get_single_covariance(self.keys[0])
        test.assert_almost_equal(cov, self.prior_factor.covariance)

        # add and init lmk
        flag_ambiguous_lmk, flag_uninit_lmk = solver.add_factor(self.range_factor1)
        self.assertTrue(flag_uninit_lmk)
        self.assertTrue(flag_ambiguous_lmk)
        # the range factor won't be added into isam since it will cause exception
        # we mainly test how we handle the exception of indeterminant systems
        self.assertEqual(solver.gaussian_solver.get_factor_size(), 2)
        self.assertEqual(solver.gaussian_solver.isam.getLinearizationPoint().size(), 1)
        self.assertEqual(set(solver.gaussian_vars), set(self.vars[:1]))
        self.assertEqual(solver.ambiguous_lmk, [self.vars[-1]])
        self.assertTrue(len(solver.uninit_lmk)==0 and
                        len(solver.lmk2new_factors)==1 and
                        len(solver.lmk2old_factors)==0 and
                        len(solver.lmk2factors)==1 and
                        len(solver.lmk2isam_factor_indices)==0)
        self.assertEqual(len(solver.cached_lmk2values), 1)
        self.assertEqual(len(solver.cached_lmk2graph), 1)
        solver.update_lmk_belief()
        reinit_lmk = solver.reinit_check()
        self.assertTrue(len(reinit_lmk) == 0)
        self.assertTrue(len(solver.uninit_lmk)==0 and
                        len(solver.lmk2new_factors)==0 and
                        len(solver.lmk2old_factors)==1 and
                        len(solver.lmk2factors)==1)
        samples, vars = solver.posterior_samples()
        self.assertEqual(set(vars), {self.vars[0], self.vars[-1]})
        if show_plot:
            plot_2d_samples(samples_array=samples, variable_ordering=vars, show_plot=show_plot)

        # add odometry
        flag_ambiguous_lmk, flag_uninit_lmk = solver.add_factor(self.odom_factor1)
        self.assertFalse(flag_uninit_lmk)
        self.assertFalse(flag_ambiguous_lmk)
        self.assertEqual(solver.gaussian_solver.get_factor_size(), 3)
        init_pose = solver.gaussian_solver.isam.getLinearizationPoint().atPose2(self.keys[1])
        test.assert_almost_equal([init_pose.x(), init_pose.y(), init_pose.theta()], self.arrays[1])
        self.assertTrue(len(solver.lmk2new_factors)==0 and
                        len(solver.lmk2old_factors)==1 and
                        len(solver.lmk2factors)==1)
        self.assertEqual(set(solver.gaussian_vars), set(self.vars[:2]))
        est_pose = solver.gaussian_solver.get_point_estimate(self.keys[1], self.vars[1].__class__.__name__)
        test.assert_almost_equal([est_pose.x(), est_pose.y(), est_pose.theta()], self.arrays[1])
        solver.update_lmk_belief()
        reinit_lmk = solver.reinit_check()
        self.assertTrue(len(reinit_lmk) == 0)
        self.assertEqual(len(solver.cached_lmk2values), 1)
        self.assertEqual(len(solver.cached_lmk2graph), 1)

        samples, vars = solver.posterior_samples()
        self.assertEqual(set(vars), set(self.vars[:2] + self.vars[-1:]))
        if show_plot:
            plot_2d_samples(samples_array=samples, variable_ordering=vars, show_plot=show_plot)

        # add and init lmk
        flag_ambiguous_lmk, flag_uninit_lmk = solver.add_factor(self.range_factor2)
        self.assertFalse(flag_uninit_lmk)
        self.assertTrue(flag_ambiguous_lmk)
        # at this time, the new factor can lead to a determined linear system
        # so the cached factors will be successfully added to the system
        self.assertEqual(solver.gaussian_solver.get_factor_size(), 5)
        self.assertEqual(solver.gaussian_solver.isam.getLinearizationPoint().size(), 3)
        self.assertEqual(set(solver.gaussian_vars), set(self.vars[:2] + self.vars[-1:]))
        self.assertEqual(solver.ambiguous_lmk, [self.vars[-1]])
        self.assertTrue(len(solver.uninit_lmk)==0 and
                        len(solver.lmk2new_factors[self.vars[-1]])==1 and
                        len(solver.lmk2old_factors[self.vars[-1]])==1 and
                        len(solver.lmk2factors[self.vars[-1]])==2 and
                        len(solver.lmk2isam_factor_indices[self.vars[-1]])==2)
        self.assertEqual(len(solver.cached_lmk2values), 0)
        self.assertEqual(len(solver.cached_lmk2graph), 0)
        lmk_pt1 = solver.gaussian_solver.get_point_estimate(self.keys[-1], self.vars[-1].__class__.__name__)
        print("isam landmark estimate after two range factors: ", lmk_pt1)
        # self.assertTrue(min(np.linalg.norm(np.array([[0,2],[0,-2]]) - lmk_pt1, axis = 1)) < .1)
        solver.update_lmk_belief()
        samples, vars = solver.posterior_samples()
        if show_plot:
            plot_2d_samples(samples_array=samples, variable_ordering=vars, show_plot=show_plot)

        reinit_lmk = solver.reinit_check()
        if len(reinit_lmk) > 0:
            solver.gaussian_solver.update_estimates()
            print("new linearization point: ", solver.gaussian_solver.isam.getLinearizationPoint().atPoint2(self.keys[-1]))
            lmk_pt1 = solver.gaussian_solver.get_point_estimate(self.keys[-1], self.vars[-1].__class__.__name__)
            print("isam landmark estimate after reinit: ", lmk_pt1)
            self.assertTrue(min(np.linalg.norm(np.array([[0,2],[0,-2]]) - lmk_pt1, axis = 1)) < .5)
            samples, vars = solver.posterior_samples()
            if show_plot:
                plot_2d_samples(samples_array=samples, variable_ordering=vars, show_plot=show_plot)

        # add and init lmk
        flag_ambiguous_lmk, flag_uninit_lmk = solver.add_factor(self.odom_factor2)
        # note that
        flag_ambiguous_lmk, flag_uninit_lmk = solver.add_factor(self.range_factor3)
        self.assertFalse(flag_uninit_lmk)
        self.assertTrue(flag_ambiguous_lmk)
        # at this time, the new factor can lead to a determined linear system
        # so the cached factors will be successfully added to the system
        self.assertEqual(solver.gaussian_solver.isam.getLinearizationPoint().size(), 4)
        self.assertEqual(set(solver.gaussian_vars), set(self.vars))
        self.assertEqual(solver.ambiguous_lmk, [self.vars[-1]])
        self.assertTrue(len(solver.uninit_lmk)==0 and
                        len(solver.lmk2new_factors[self.vars[-1]])==1 and
                        len(solver.lmk2old_factors[self.vars[-1]])==2 and
                        len(solver.lmk2factors[self.vars[-1]])==3 and
                        len(solver.lmk2isam_factor_indices[self.vars[-1]])==3)
        self.assertEqual(len(solver.cached_lmk2values), 0)
        self.assertEqual(len(solver.cached_lmk2graph), 0)
        lmk_pt1 = solver.gaussian_solver.get_point_estimate(self.keys[-1], self.vars[-1].__class__.__name__)
        print("isam landmark estimate after three range factors: ", lmk_pt1)
        # self.assertTrue(min(np.linalg.norm(np.array([[0,2],[0,-2]]) - lmk_pt1, axis = 1)) < .1)
        solver.update_lmk_belief()
        samples, vars = solver.posterior_samples()
        if show_plot:
            plot_2d_samples(samples_array=samples, variable_ordering=vars, show_plot=show_plot)

        reinit_lmk = solver.reinit_check()
        if len(reinit_lmk) > 0:
            solver.gaussian_solver.update_estimates()
            print("new linearization point: ", solver.gaussian_solver.isam.getLinearizationPoint().atPoint2(self.keys[-1]))
            lmk_pt1 = solver.gaussian_solver.get_point_estimate(self.keys[-1], self.vars[-1].__class__.__name__)
            print("isam landmark estimate after reinit: ", lmk_pt1)
            self.assertTrue(min(np.linalg.norm(np.array([[0,2],[0,-2]]) - lmk_pt1, axis = 1)) < 1.0)
            samples, vars = solver.posterior_samples()
            if show_plot:
                plot_2d_samples(samples_array=samples, variable_ordering=vars, show_plot=show_plot)

if __name__ == '__main__':
    unittest.main()