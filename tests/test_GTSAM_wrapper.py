import unittest
import numpy.testing as test
from adaptive_inference.GTSAM_help import *
from adaptive_inference.GaussianSolverWrapper import gtsamWrapper
import time

from adaptive_inference.utils import to_Key
from utils.Visualization import plot_2d_samples


class gtsamWrapperTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vars = [SE2Variable('X0'), SE2Variable('X1'), SE2Variable('X2'), R2Variable('L1', variable_type=VariableType.Landmark)]
        cls.keys = [to_Key(var) for var in cls.vars]
        cls.varstype = [var.__class__.__name__ for var in cls.vars]

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

        cls.groundtruth = (gtsam.Pose2(*[.0,.0,.0]), gtsam.Pose2(*[2.0,.0,np.pi/2]), gtsam.Pose2(*[2.0, 2.0, np.pi/2]), gtsam.Point2(*[.0, 2.0]))
        cls.lmk_factor_indices = [3,4,5]
        cls.gtsam_factors = [to_gtsam_factor(f) for f in cls.factors]

    def setupSolver(self):
        solver = gtsamWrapper()
        values = gtsam.Values()
        for i in range(len(self.groundtruth)):
            values.insert(self.keys[i], self.groundtruth[i])
        factor_graph = gtsam.NonlinearFactorGraph()
        for f in self.gtsam_factors:
            factor_graph.add(f)
        solver.update_factor_value(factor_graph, values)
        return solver

    def test_update_factor_value(self):
        solver = self.setupSolver()
        isam = solver.isam
        self.assertEqual(isam.getFactorsUnsafe().size(), len(self.factors))
        test.assert_almost_equal(isam.getLinearizationPoint().atPoint2(self.keys[-1]), [0, 2])
        pose = isam.getLinearizationPoint().atPose2(self.keys[-2])
        test.assert_almost_equal([pose.x(), pose.y(), pose.theta()], [2, 2, np.pi / 2])

    def test_reinitialize(self):
        solver = self.setupSolver()
        new_indices = solver.reinitialize(self.keys[-1], gtsam.Point2(0, -2), self.lmk_factor_indices)
        isam = solver.isam
        self.assertEqual(isam.getFactorsUnsafe().size(), len(self.factors) + len(self.lmk_factor_indices))
        test.assert_almost_equal(new_indices, len(self.factors) + np.arange(len(self.lmk_factor_indices)))
        test.assert_almost_equal(isam.getLinearizationPoint().atPoint2(self.keys[-1]), [0, -2])

    def test_estimate(self):
        solver = self.setupSolver()
        solver.update_estimates()
        means = []
        covs = []

        # test mean
        for i in range(len(self.keys)):
            means.append(solver.get_point_estimate(self.keys[i], self.varstype[i]))
            covs.append(solver.get_single_covariance(self.keys[i]))
            if self.varstype[i][:2] == 'SE':
                test.assert_almost_equal(means[i].matrix(), self.groundtruth[i].matrix())
            else:
                test.assert_almost_equal(means[i], self.groundtruth[i])
        # print(covs)

        # test covariance
        new_vartypes = [self.varstype[self.keys.index(v)] for v in solver.var_ordering]
        joint_cov = solver.get_joint_covariance(list(solver.var_ordering), new_vartypes)
        test.assert_almost_equal(joint_cov, solver._marginals.jointMarginalCovariance(solver.var_ordering).fullMatrix())
        joint_cov = solver.get_joint_covariance(self.keys, self.varstype)
        joint_cov2 = solver.get_joint_covariance(self.keys[::-1], self.varstype[::-1])
        # print(joint_cov)
        # print(joint_cov2)
        test.assert_almost_equal(joint_cov[0: self.vars[0].dim, self.vars[0].dim: self.vars[0].dim + self.vars[1].dim],
                               joint_cov2[-self.vars[0].dim:, -self.vars[0].dim - self.vars[1].dim: -self.vars[0].dim])

        test.assert_almost_equal(joint_cov[0: self.vars[0].dim, 0: self.vars[0].dim],
                               joint_cov2[-self.vars[0].dim:, -self.vars[0].dim:])
        test.assert_almost_equal(solver.get_joint_covariance(self.keys[:-1]), joint_cov[:-self.vars[-1].dim, :-self.vars[-1].dim])

        # test sampling
        start = time.time()
        samples = solver.get_samples(self.keys, self.varstype, 200)
        end = time.time()
        print("Elapsed time: ", -start + end)
        plot_2d_samples(samples_array=samples, variable_ordering=self.vars, show_plot=True)

if __name__ == '__main__':
    unittest.main()