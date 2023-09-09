import unittest
from geometry.TwoDimension import Point2, Rot2, SE2Pose
import numpy as np

class TestCase(unittest.TestCase):
    def test_point2(self) -> None:
        #constructor
        p0 = Point2()
        px = Point2(x = 1.0)
        py = Point2(y = 1.0)
        p1 = Point2(x = 1.0, y = 1.0)
        self.assertEqual(p0, Point2(0,0))
        self.assertEqual(px, Point2(1,0))
        self.assertEqual(py, Point2(0,1))
        self.assertEqual(p1, Point2(1,1))

        #instance methods
        self.assertEqual(p1.x, 1)
        self.assertEqual(p1.y, 1)
        self.assertEqual(p1.dim, 2)
        self.assertEqual(p1, p1.copy())
        self.assertFalse(id(p1) == id(p1.copy()))
        self.assertEqual(p1.inverse(), Point2(-1,-1))
        self.assertEqual(p1.norm, np.sqrt(2))
        self.assertEqual(p0.distance(p1), np.sqrt(2))
        self.assertEqual(p0.transform_to(p1), p1 - p0)

        #magic methods for arithmetic operations
        self.assertEqual(px+py, p1)
        self.assertEqual(p1-px, py)
        self.assertEqual(p1*0.0, p0)
        self.assertEqual(p1*1.0, p1)


        p_tmp = Point2()
        p_id_prev = id(p_tmp)
        p_tmp += p1
        self.assertEqual(p_tmp, p1)
        p_tmp -= px
        self.assertEqual(p_tmp, py)
        p_tmp *= 0.5
        self.assertEqual(p_tmp, Point2(0,0.5))
        p_tmp /= 0.5
        self.assertEqual(p_tmp, Point2(0, 1))
        self.assertEqual(p_id_prev, id(p_tmp))
        self.assertEqual(-p1, Point2(-1,-1))
        print("p1 is: ", p1)

    def test_rot2(self) -> None:
        #constructor
        r0 = Rot2()
        r45 = Rot2(np.pi / 4)
        r45_deg = Rot2.by_degrees(45)
        r45_xy = Rot2.by_xy(1, 1)
        r45_matrix = Rot2.by_matrix(np.array([ [0.5, -0.5],
                                               [0.5,  0.5] ]))
        self.assertTrue(r45 == r45_deg and
                        r45 == r45_matrix and
                        r45 == r45_xy)
        self.assertEqual(r0.dim, 1)
        self.assertEqual(r45.degrees, 45)
        self.assertEqual(r45.theta, np.pi / 4)

        self.assertIsNone(
            np.testing.assert_almost_equal(r45.matrix,
                                           np.array([ [1/np.sqrt(2), -1/np.sqrt(2)],
                                                      [1/np.sqrt(2),  1/np.sqrt(2)] ])))
        self.assertAlmostEqual(r45.cos, 1/np.sqrt(2))
        self.assertAlmostEqual(r45.sin, 1/np.sqrt(2))
        self.assertIsNone(np.testing.assert_almost_equal(
                                r45.log_map(),
                                np.array([np.pi / 4])))
        self.assertEqual(r45.inverse(),
                         Rot2(-np.pi/4 + 4 * np.pi))
        self.assertEqual(r45, r45.copy())

        pt = Point2(1/np.sqrt(2),
                    1/np.sqrt(2))
        pt_rot = r45.rotate_point(pt)
        self.assertTrue(pt_rot == Point2(0.0,1.0))
        pt_unrot = r45.unrotate_point(pt)
        self.assertTrue(pt_unrot == Point2(1.0, 0.0))
        self.assertTrue(r45 * r45 == Rot2.by_degrees(90))
        self.assertTrue(r45 * r45 * pt == Point2(-1/np.sqrt(2),
                                                1/np.sqrt(2)))
        self.assertEqual(r45 / r45, r0)

        p_tmp = Rot2()
        p_id_prev = id(p_tmp)
        p_tmp *= r45
        self.assertEqual(p_tmp, r45)
        p_tmp /= r45
        self.assertEqual(p_tmp, r0)
        self.assertEqual(p_id_prev, id(p_tmp))
        print("r45 is: ", r45)

    def test_pose2(self) -> None:
        #constructor
        r0 = SE2Pose()
        th45 = np.pi/4
        x = 1.0
        y = 1.0
        r45 = SE2Pose(x, y, th45)
        r45_pt_rt = SE2Pose.by_pt_rt(pt = Point2(x, y), rt = Rot2(th45))
        r45_matrix = SE2Pose.by_matrix(np.array([[np.cos(th45), -np.sin(th45), x],
                                                 [np.sin(th45),  np.cos(th45), y],
                                                 [0.0,  0.0, 1.0]]))
        J = np.array([ [np.sin(th45), (np.cos(th45) - 1)],
                       [(1 - np.cos(th45)), np.sin(th45)]])/th45
        t_delta = np.linalg.inv(J) @ np.array([x,y]).transpose()
        manifold_delta = np.zeros(3)
        manifold_delta[0:2] = t_delta[:]
        manifold_delta[2] = th45
        r45_exp = SE2Pose.by_exp_map(vector = manifold_delta)
        
        self.assertTrue(r45 == r45_pt_rt and
                        r45 == r45_matrix and
                        r45 == r45_exp)
        self.assertEqual(r45.dim, 3)

        self.assertEqual(r45.theta, th45)
        self.assertEqual(r45.x, x)
        self.assertEqual(r45.y, y)
        self.assertTrue(r45.rotation == Rot2(th45))
        self.assertTrue(r45.translation == Point2(x,y))

        self.assertIsNone(
            np.testing.assert_almost_equal(r45.matrix,
                                           np.array([ [1/np.sqrt(2),  -1/np.sqrt(2), x],
                                                      [1/np.sqrt(2),   1/np.sqrt(2), y],
                                                      [           0,              0, 1]
                                                    ])
                                           )
        )

        log_map = r45.log_map()
        self.assertIsNone(
            np.testing.assert_almost_equal(log_map,
                                           manifold_delta))

        pt = Point2(0,1)
        range, bearing = r45.range_and_bearing(pt)
        self.assertEqual(range, 1)
        self.assertEqual(bearing, 3 * np.pi / 4)
        self.assertEqual(r45.inverse(), SE2Pose(x = -np.sqrt(2), y = 0, theta = -th45))
        self.assertEqual(r45, r45.copy())
        self.assertEqual(r45.transform_to(r0), r45.inverse())

        pt_world_rf = r45.transform_point(Point2(0, np.sqrt(2)))
        self.assertEqual(pt_world_rf,
                         Point2(0,2))

        self.assertTrue(r45 * r45 == SE2Pose(x = 1, y =1 + np.sqrt(2), theta =np.pi / 2))

        self.assertTrue(r45 / r45 == r0)

        p_tmp = SE2Pose()
        p_id_prev = id(p_tmp)
        p_tmp *= r45
        self.assertEqual(p_tmp, r45)
        p_tmp /= r45
        self.assertEqual(p_tmp, r0)
        self.assertEqual(p_id_prev, id(p_tmp))
        print("pose45 is: ", r45)

if __name__ == '__main__':
    unittest.main()
