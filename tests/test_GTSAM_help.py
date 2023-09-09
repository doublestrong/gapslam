import unittest

import numpy.testing as test

from adaptive_inference.GaussianSolverWrapper import gtsam_reinit
from adaptive_inference.GTSAM_help import *
from adaptive_inference.utils import to_Key


class GTSAMhelpTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vars = [Variable('X1', 3), Variable('X2', 3), Variable('L1', 2, variable_type=VariableType.Landmark)]
        cls.values = ([0,5.0,-np.pi/2],[0., 10.0, -np.pi/2], [10.0, 5.0])
        cls.prior_factor = UnarySE2ApproximateGaussianPriorFactor(var=cls.vars[0],
                                                                  prior_pose=SE2Pose(0, 5.0, np.pi/2),
                                                                  covariance=np.diag([3,2,1]))
        cls.odom_factor = SE2RelativeGaussianLikelihoodFactor(var1=cls.vars[0],
                                                              var2=cls.vars[1],
                                                              observation=SE2Pose(5.0, .0, .0),
                                                              covariance=np.diag([3,2,1]))
        cls.range_factor = SE2R2RangeGaussianLikelihoodFactor(var1=cls.vars[0],
                                                              var2=cls.vars[2],
                                                              observation=10.0,
                                                              sigma=3.0
                                                              )
        cls.range_factor2 = SE2R2RangeGaussianLikelihoodFactor(var1=cls.vars[1],
                                                              var2=cls.vars[2],
                                                              observation=13.0,
                                                              sigma=3.0)
        cls.factors = [cls.prior_factor, cls.range_factor, cls.range_factor2, cls.odom_factor]

    def test_vars(self):
        gtsam_vars = [to_Key(var) for var in self.vars]
        print([gtsam.Symbol(var).string() for var in gtsam_vars])
        self.assertEqual(gtsam_vars[0], gtsam.Symbol('X', 1).key())
        test.assert_equal(gtsam_vars[1], gtsam.Symbol('X', 2).key())
        test.assert_equal(gtsam_vars[2], gtsam.Symbol('L', 1).key())

    def test_prior_pose2(self):
        gtsam_f = to_gtsam_factor(self.prior_factor)
        self.assertEqual(gtsam_f.dim(), 3)
        test.assert_array_equal(gtsam_f.noiseModel().sigmas(), np.sqrt([3,2,1]))
        # size refers to the number of connected variable nodes
        self.assertEqual(gtsam_f.size(), 1)
        self.assertEqual([gtsam_f.prior().x(), gtsam_f.prior().y(), gtsam_f.prior().theta()], [0, 5.0, np.pi/2])

    def test_between_pose2(self):
        gtsam_f = to_gtsam_factor(self.odom_factor)
        self.assertEqual(gtsam_f.dim(), 3)
        test.assert_array_equal(gtsam_f.noiseModel().sigmas(), np.sqrt([3,2,1]))
        self.assertEqual(gtsam_f.size(), 2)
        self.assertEqual([gtsam_f.measured().x(), gtsam_f.measured().y(), gtsam_f.measured().theta()], [5.0, .0, .0])

    def test_range_2d(self):
        gtsam_f = to_gtsam_factor(self.range_factor)
        # dim of a factor is the dimension of the measured value
        self.assertEqual(gtsam_f.dim(), 1)
        test.assert_array_equal(gtsam_f.noiseModel().sigmas(), np.array([3]))
        self.assertEqual(gtsam_f.size(), 2)
        # self.assertEqual([gtsam_f.measured().x(), gtsam_f.measured().y(), gtsam_f.measured().theta()], [5.0, .0, .0])

    def test_values_isam_graph_reinit(self):
        graph = gtsam.NonlinearFactorGraph()
        for f in self.factors:
            graph.add(to_gtsam_factor(f))
        self.assertEqual(graph.size(), len(self.factors))

        values = gtsam.Values()
        gt_vars = []
        for i, var in enumerate(self.vars):
            gt_vars.append(to_Key(var))
            if var.dim == 2:
                values.insert(gt_vars[i], gtsam.Point2(*self.values[i]))
            else:
                values.insert(gt_vars[i], gtsam.Pose2(*self.values[i]))

        isam = gtsam.ISAM2()
        isam.update(graph, values)

        lmk_var = gt_vars[2]
        self.assertEqual(gtsam.Symbol(lmk_var).string(), 'L1')
        lmk2idx = [1, 2]

        lmk2idx = gtsam_reinit(isam, lmk_var, lmk2idx, gtsam.Point2(5, 10.0))
        tmp_graph = isam.getFactorsUnsafe()
        self.assertEqual(tmp_graph.size(), 6)
        self.assertEqual(lmk2idx, [4, 5])

        # check linearization point
        test.assert_almost_equal(isam.getLinearizationPoint().atPoint2(lmk_var), np.array([5,10]))

if __name__ == '__main__':
    unittest.main()