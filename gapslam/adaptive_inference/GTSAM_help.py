from adaptive_inference.GaussianSolverWrapper import GaussianSolverWrapper
from factors.Factors import *
from slam.Variables import *

def to_PriorFactorPose2(f: UnarySE2ApproximateGaussianPriorFactor):
    var1 = f.vars[0].key
    abs_pose2 = gtsam.Pose2(*f.observation)
    pose2_nm = gtsam.noiseModel.Gaussian.Covariance(f.covariance)
    return gtsam.PriorFactorPose2(var1, abs_pose2, pose2_nm)

def to_PriorFactorPoint2(f: UnaryR2GaussianPriorFactor):
    var1 = f.vars[0].key
    abs_point2 = gtsam.Point2(*f.observation)
    point2_nm = gtsam.noiseModel.Gaussian.Covariance(f.covariance)
    return gtsam.PriorFactorPoint2(var1, abs_point2, point2_nm)

def to_BetweenFactorPose2(f: SE2RelativeGaussianLikelihoodFactor):
    var1 = f.vars[0].key
    var2 = f.vars[1].key
    rel_pose2 = gtsam.Pose2(*f.observation)
    # note that for 2D cases in both GTSAM and NFiSAM, the diagonal entries in the covariance correspond to x,
    # y and theta
    pose2_nm = gtsam.noiseModel.Gaussian.Covariance(f.covariance)
    return gtsam.BetweenFactorPose2(var1, var2, rel_pose2, pose2_nm)

def to_RangeFactor2D(f: SE2R2RangeGaussianLikelihoodFactor):
    var1 = f.vars[0].key
    var2 = f.vars[1].key
    range = f.observation[0] # the default return is np.ndarray
    # note that in NFiSAM, the sigma for the range factor is the standard deviation
    # in the error of range measurements
    range_nm = gtsam.noiseModel.Isotropic.Sigma(1, f.sigma)
    return gtsam.RangeFactor2D(var1, var2, range, range_nm)

def to_NullHypo(f: BinaryFactorWithNullHypo):
    if isinstance(f.components[0], SE2R2RangeGaussianLikelihoodFactor):
        return to_RangeFactor2D(f.components[0])
    else:
        raise NotImplementedError

def to_BearingFactorPose2(f: SE2BearingLikelihoodFactor):
    """
    convert to a bearing factor between two poses
    """
    var1 = f.vars[0].key
    var2 = f.vars[1].key
    b = f.observation[0] # the default return is np.ndarray
    # note that in NFiSAM, the sigma for the bearing factor is the standard deviation
    # in the error of range measurements
    b_nm = gtsam.noiseModel.Isotropic.Sigma(1, f.sigma)
    return gtsam.BearingFactorPose2(var1, var2, gtsam.Rot2(b), b_nm)

def to_BearingFactor2D(f: SE2BearingLikelihoodFactor):
    """
    convert to a bearing factor between a pose2 and a point2
    """
    var1 = f.vars[0].key
    var2 = f.vars[1].key
    b = f.observation[0] # the default return is np.ndarray
    # note that in NFiSAM, the sigma for the bearing factor is the standard deviation
    # in the error of range measurements
    b_nm = gtsam.noiseModel.Isotropic.Sigma(1, f.sigma)
    return gtsam.BearingFactor2D(var1, var2, gtsam.Rot2(b), b_nm)

def to_CameraProjectionFactor(f: CameraProjectionFactor):
    """
    convert to a camera projection factor between a pose and a point
    """
    var1 = f.vars[0].key
    var2 = f.vars[1].key
    cam_model = gtsam.Cal3_S2(f.fx, f.fy, .0, f.cx,
                              f.cy)
    cam_noise = gtsam.noiseModel.Gaussian.Covariance(f.covariance)

    return gtsam.GenericProjectionFactorCal3_S2(
        f.observation, cam_noise, var1, var2, cam_model)

def to_PriorFactorPose3(f: UnarySE3Factor):
    var1 = f.vars[0].key
    abs_pose2 = gtsam.Pose3(f.observation)
    pose3_nm = gtsam.noiseModel.Gaussian.Covariance(f.covariance)
    return gtsam.PriorFactorPose3(var1, abs_pose2, pose3_nm)

def to_BetweenFactorPose3(f: SE3RelativeGaussianLikelihoodFactor):
    var1 = f.vars[0].key
    var2 = f.vars[1].key
    rel_pose3 = gtsam.Pose3(f.observation)
    # note that for 3D cases in GTSAM, the diagonal entries in the covariance correspond to rotation and translation
    pose3_nm = gtsam.noiseModel.Gaussian.Covariance(f.covariance)
    return gtsam.BetweenFactorPose3(var1, var2, rel_pose3, pose3_nm)

# converting our own factors to GTSAM factors
# TODO: maybe add to_gtsam to the methods in the class of factors
factor_name2function = {
    'SE2RelativeGaussianLikelihoodFactor': to_BetweenFactorPose2,
    'UnarySE2ApproximateGaussianPriorFactor': to_PriorFactorPose2,
    'UnaryR2GaussianPriorFactor': to_PriorFactorPoint2,
    'SE2R2RangeGaussianLikelihoodFactor': to_RangeFactor2D,
    'SE2BearingLikelihoodFactor': to_BearingFactorPose2,
    'SE2R2BearingLikelihoodFactor': to_BearingFactor2D,
    'BinaryFactorWithNullHypo': to_NullHypo,
    'CameraProjectionFactor': to_CameraProjectionFactor,
    'SE3RelativeGaussianLikelihoodFactor': to_BetweenFactorPose3,
    'UnarySE3Factor': to_PriorFactorPose3
}

def to_gtsam_factor(f: Factor):
    """
    Translate our own factors to GTSAM factors
    param:
        f: fields in f will be used to create GTSAM factors
    """
    class_name = f.__class__.__name__
    return factor_name2function[class_name](f)

def init_BetweenPose2(factor: SE2RelativeGaussianLikelihoodFactor, gs_solver: GaussianSolverWrapper, old_var: Variable):
    old_pose = gs_solver.get_point_estimate(old_var.key, vartype="SE2Variable")
    if old_var == factor.var1:
        init_val = old_pose * gtsam.Pose2(*factor.observation)
    else:
        init_val = old_pose * gtsam.Pose2(*factor.observation).inverse()
    return init_val

def init_PriorPose2(factor: UnarySE2ApproximateGaussianPriorFactor, gs_solver: GaussianSolverWrapper = None, old_var: SE2Variable = None):
    return gtsam.Pose2(*factor.observation)

def init_Range2D(factor: SE2R2RangeGaussianLikelihoodFactor, gs_solver: GaussianSolverWrapper, old_var: Variable):
    if isinstance(old_var, SE2Variable):
        old_pose = gs_solver.get_point_estimate(old_var.key, vartype="SE2Variable")
        range = factor.observation[0]
        bearing = np.random.rand() * 2 * np.pi
        new_x = old_pose.x() + range * np.cos(bearing)
        new_y = old_pose.y() + range * np.sin(bearing)
        init_val = gtsam.Point2(new_x, new_y)
    else:
        old_point = gs_solver.get_point_estimate(old_var.key, vartype="R2Variable")
        range = factor.observation[0]
        bearing = np.random.rand() * 2 * np.pi
        orient = np.random.rand() * 2 * np.pi
        new_x = old_point[0] + range * np.cos(bearing)
        new_y = old_point[0] + range * np.sin(bearing)
        init_val = gtsam.Pose2(new_x, new_y, orient)
    return init_val

def init_mixtures(factor: BinaryFactorWithNullHypo, gs_solver: GaussianSolverWrapper, old_var: Variable):
    if isinstance(factor.components[0], SE2R2RangeGaussianLikelihoodFactor):
        return init_Range2D(factor.components[0], gs_solver, old_var)
    else:
        raise NotImplementedError

def init_Bearing2D(factor: SE2R2BearingLikelihoodFactor, gs_solver: GaussianSolverWrapper, old_var: Variable):
    if isinstance(old_var, SE2Variable):
        old_pose = gs_solver.get_point_estimate(old_var.key, vartype="SE2Variable")
        range = np.random.rand() * (factor._max_range - factor._min_range) + factor._min_range
        bearing = factor.observation[0]
        new_x = old_pose.x() + range * np.cos(bearing + old_pose.theta())
        new_y = old_pose.y() + range * np.sin(bearing + old_pose.theta())
        init_val = gtsam.Point2(new_x, new_y)
    else:
        old_point = gs_solver.get_point_estimate(old_var.key, vartype="R2Variable")
        range = np.random.rand() * (factor._max_range - factor._min_range) + factor._min_range
        bearing = factor.observation[0]
        trans_rot = np.random.rand() * 2 * np.pi
        new_x = old_point.x() + range * np.cos(trans_rot)
        new_y = old_point.y() + range * np.sin(trans_rot)
        init_val = gtsam.Pose2(new_x, new_y, trans_rot - bearing)
    return init_val

def init_CameraProjection(factor: CameraProjectionFactor, gs_solver: GaussianSolverWrapper, old_var: Variable):
    assert isinstance(old_var, SE3Variable)
    old_pose = gs_solver.get_point_estimate(old_var.key, vartype="SE3Variable")
    depth = np.random.rand() * (factor._max_depth - factor._min_depth) + factor._min_depth
    pt = SE3Pose(old_pose.matrix()).depth2point(factor.observation, depth,factor.cam_intrinsic_mat)
    init_val = gtsam.Point3(pt[0], pt[1], pt[2])
    return init_val

def init_BetweenPose3(factor: SE3RelativeGaussianLikelihoodFactor, gs_solver: GaussianSolverWrapper, old_var: Variable):
    old_pose = gs_solver.get_point_estimate(old_var.key, vartype="SE3Variable")
    if old_var == factor.var1:
        init_val = old_pose * gtsam.Pose3(factor.observation)
    else:
        init_val = old_pose * gtsam.Pose3(factor.observation).inverse()
    return init_val

def init_PriorPose3(factor: UnarySE3Factor, gs_solver: GaussianSolverWrapper = None, old_var: SE3Variable = None):
    return gtsam.Pose3(factor.observation)

boostrap_init_factors = {'SE2RelativeGaussianLikelihoodFactor': init_BetweenPose2,
                 'UnarySE2ApproximateGaussianPriorFactor': init_PriorPose2,
                 'SE2R2RangeGaussianLikelihoodFactor': init_Range2D,
                 'SE2R2BearingLikelihoodFactor': init_Bearing2D,
                 'BinaryFactorWithNullHypo': init_mixtures,
                 'CameraProjectionFactor': init_CameraProjection,
                 'SE3RelativeGaussianLikelihoodFactor': init_BetweenPose3,
                 'UnarySE3Factor': init_PriorPose3}


def init_gtsam_var(f: Factor, gs: GaussianSolverWrapper, old_var: Variable):
    """
    Translate our own factors to GTSAM factors
    param:
        f: fields in f will be used to create GTSAM value
        gs: gs provides current estimate
        old_var: variable that has been initialized
    """
    class_name = f.__class__.__name__
    if class_name in boostrap_init_factors:
        return boostrap_init_factors[class_name](f, gs, old_var)
    else:
        raise NotImplementedError(f"Unknown factor class {class_name} for bootstrapped initialization")

def reg_factor(key: "GTSAM Key", values: "GTSAM Values", cov_scale: int, var: Variable):
    """
    Create a prior factor for var with the given scale of covariance
    param:
        key: key
        values: values
        cov_scale: int
        var: our own variable
    """
    if isinstance(var, SE2Variable):
        pose2_nm = gtsam.noiseModel.Isotropic.Sigma(3, cov_scale)
        return gtsam.PriorFactorPose2(key, values, pose2_nm)
    elif isinstance(var, R2Variable):
        point2_nm = gtsam.noiseModel.Isotropic.Sigma(2, cov_scale)
        return gtsam.PriorFactorPoint2(key, values, point2_nm)
    elif isinstance(var, R3Variable):
        point3_nm = gtsam.noiseModel.Isotropic.Sigma(3, cov_scale)
        return gtsam.PriorFactorPoint3(key, values, point3_nm)
    elif isinstance(var, SE3Variable):
        pose3_nm = gtsam.noiseModel.Isotropic.Sigma(6, cov_scale)
        return gtsam.PriorFactorPose3(key, values, pose3_nm)
    else:
        raise NotImplementedError(f"Unknown variable class for regularization factors.")