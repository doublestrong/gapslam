from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np

from factors.Factors import Factor, BinaryFactor, UnaryFactor, AmbiguousDataAssociationFactor, PriorFactor, \
    SE2RelativeGaussianLikelihoodFactor
from geometry.TwoDimension import SE2Pose
from slam.FactorGraphSimulator import read_factor_graph_from_file, G2oToroPoseGraphReader
from slam.Variables import Variable, VariableType
from utils.Functions import sort_pair_lists

def group_nodes_factors_incrementally_old(
        nodes: List[Variable], factors: List[Factor],
        incremental_step: int = None,
) -> List[Tuple[List[Variable], List[Factor]]]:
    """
    The function groups nodes and factors in an incremental way
    It is assumed that factors are all unary, binary, and k-way factors
    :param nodes
        The temporal ordering of nodes are decided by their indices in the list
    :param factors
    :incremental_step: the number of poses for each incremental update
        when set to None, computation should be performed in a batch mode
    :return: List
    """
    if incremental_step is None:
        incremental_step = len(nodes)
    remaining_nodes = nodes.copy()
    remaining_factors = factors.copy()
    res = []
    while remaining_nodes:
        print("Number of Remaining nodes: ", len(remaining_nodes))
        new_factors = []
        new_nodes = []
        remaining_poses = [node for node in remaining_nodes
                           if node.type == VariableType.Pose]
        for i in range(incremental_step):
            if remaining_poses:
                node = remaining_poses.pop(0)
                new_nodes.append(node)
                remaining_nodes.remove(node)
            else:
                break

        # Check for likelihood factors
        for factor in remaining_factors:
            if not isinstance(factor, (BinaryFactor, AmbiguousDataAssociationFactor)):
                continue
            if not set.intersection(set(remaining_nodes), set(factor.vars)):
                # this is for binary loop closure and data associations
                new_factors.append(factor)
            elif isinstance(factor, BinaryFactor):
                var1, var2 = factor.vars
                if var1 not in remaining_nodes and var2.type == \
                        VariableType.Landmark:
                    new_factors.append(factor)
                    remaining_nodes.remove(var2)
                    new_nodes.append(var2)
                elif var2 not in remaining_nodes and var1.type == \
                        VariableType.Landmark:
                    new_factors.append(factor)
                    remaining_nodes.remove(var1)
                    new_nodes.append(var1)

        # Check for unary prior factors and AmbiguousDataAssociationFactor
        # This can only be done after landmarks are added
        for factor in remaining_factors:
            if not isinstance(factor, UnaryFactor):
                continue
            if not set.intersection(set(remaining_nodes), set(factor.vars)):
                new_factors.append(factor)

        for factor in new_factors:
            remaining_factors.remove(factor)

        res.append((new_nodes, new_factors))
    return res

def update_list_in_dict(mydict, mykey, listkey, value):
    if mykey not in mydict:
        mydict[mykey] = {listkey: [value]}
    elif listkey not in mydict[mykey]:
        mydict[mykey][listkey] = [value]
    else:
        mydict[mykey][listkey].append(value)
    return mydict


def group_nodes_factors_incrementally(
        nodes: List[Variable], factors: List[Factor],
        incremental_step: int = None,
        multirobot=True
) -> List[Tuple[List[Variable], List[Factor]]]:
    """
    The function groups nodes and factors in an incremental way
    It is assumed that factors are all unary, binary, and k-way factors
    :param nodes
        The temporal ordering of nodes are decided by their indices in the list
    :param factors
    :incremental_step: the number of poses for each incremental update
        when set to None, computation should be performed in a batch mode
    :return: List
    """
    if multirobot:
        return multirbt_group_nodes_factors_incrementally(nodes, factors, incremental_step)
    else:
        return single_robot_group_nodes_factors_incrementally(nodes, factors, incremental_step)

def single_robot_group_nodes_factors_incrementally(
        nodes: List[Variable], factors: List[Factor],
        incremental_step: int = None) -> List[Tuple[List[Variable], List[Factor]]]:
    """
    The function groups nodes and factors in an incremental way
    It is assumed that factors are all unary, binary, and k-way factors
    :param nodes
        The temporal ordering of nodes are decided by their indices in the list
    :param factors
    :incremental_step: the number of poses for each incremental update
        when set to None, computation should be performed in a batch mode
    :return: List
    """
    # index all factors and vars
    rbt_idx = []
    lmk_idx = []
    for i in range(len(nodes)):
        if nodes[i].type == VariableType.Pose:
            rbt_idx.append(i)
        else:
            lmk_idx.append(i)
    prior_idx = []
    p2p_idx = []
    p2l_idx = []
    ada_idx = []
    for i, factor in enumerate(factors):
        if isinstance(factor, UnaryFactor):
            prior_idx.append(i)
        elif isinstance(factor, BinaryFactor):
            if factor.var1.type == factor.var2.type == VariableType.Pose:
                p2p_idx.append(i)
            elif factor.var1.type == VariableType.Pose and factor.var2.type == VariableType.Landmark:
                p2l_idx.append(i)
            else:
                raise ValueError("Unknown factors: ", factor.__str__())
        elif isinstance(factor, AmbiguousDataAssociationFactor):
            ada_idx.append(i)
    # group factors and variables incrementally
    # //return values and indices of factors
    incVarFactorPairs = []
    # vector<pair<Values, vector<int>>> res;
    if (incremental_step is None or incremental_step > len(rbt_idx) or incremental_step <=0):
        incremental_step = len(rbt_idx)
        # print("Reset incremental_step to its max feasible value: ", incremental_step)

    newVars = []
    newFactors = []
    addedRbts = set()
    addedLmks = set()
    # incremental updates with robot vars
    for k, rbtid in enumerate(rbt_idx):
        rbt_node = nodes[rbtid]
        newVars.append(rbt_node)
        addedRbts.add(rbt_node)

        tmp_factor_idx = []
        for j in prior_idx:
            if factors[j].vars[0] == rbt_node:
                tmp_factor_idx.append(j)
                # print("Push a prior factor at ", rbt_node.name)
        prior_idx = [i for i in prior_idx if i not in tmp_factor_idx]
        newFactors += tmp_factor_idx

        tmp_factor_idx = []
        for j in p2p_idx:
            tmp_vars = set(factors[j].vars)
            if tmp_vars.issubset(addedRbts):
                tmp_factor_idx.append(j)
                # print("Push a p2p factor between ", factors[j].var1.name, factors[j].var2.name)
        if len(tmp_factor_idx) == 0 and len(addedRbts) > 1:
            raise ValueError("No pose2pose factors for the newly added robot variable.")
        else:
            p2p_idx = [i for i in p2p_idx if i not in tmp_factor_idx]
            newFactors += tmp_factor_idx

        tmp_factor_idx = []
        for j in p2l_idx:
            if factors[j].var1 == rbt_node:
                lmk_var = factors[j].var2
                if lmk_var not in addedLmks:
                    addedLmks.add(lmk_var)
                    newVars.append(lmk_var)
                tmp_factor_idx.append(j)
                # print("Push a landmark measurement factor between ", factors[j].var1, factors[j].var2)
        p2l_idx = [i for i in p2l_idx if i not in tmp_factor_idx]
        newFactors += tmp_factor_idx

        tmp_factor_idx = []
        for j in ada_idx:
            if factors[j].root_var == rbt_node:
                var2s = set(factors[j].child_vars)
                if not (var2s.issubset(addedRbts) or var2s.issubset(addedLmks)):
                    raise ValueError("Invalid factors: ", factors[j].__str__())
                tmp_factor_idx.append(j)
                # print("Push an ADA factor between ", factors[j].root_var, factors[j].child_vars)
        ada_idx = [i for i in ada_idx if i not in tmp_factor_idx]
        newFactors += tmp_factor_idx

        tmp_factor_idx = []
        for j in prior_idx:
            if factors[j].vars[0] in newVars:
                tmp_factor_idx.append(j)
                # print("Push a prior factor at ", rbt_node.name)
        prior_idx = [i for i in prior_idx if i not in tmp_factor_idx]
        newFactors += tmp_factor_idx

        if ((k+1) % incremental_step == 0) or k == len(rbt_idx) - 1:
            n_v = [var for var in newVars]
            n_f = [factors[j] for j in newFactors]
            incVarFactorPairs.append([n_v, n_f])
            newVars = []
            newFactors = []
            # print("New batch loaded.")
    print("There are ", len(incVarFactorPairs), " pairs of vars and factors.")
    return incVarFactorPairs

def multirbt_group_nodes_factors_incrementally(
        nodes: List[Variable], factors: List[Factor],
        incremental_step: int = None
) -> List[Tuple[List[Variable], List[Factor]]]:
    """
    The function groups nodes and factors in an incremental way
    It is assumed that factors are all unary, binary, and k-way factors
    :param nodes
        The temporal ordering of nodes are decided by their indices in the list
    :param factors
    :incremental_step: the number of poses for each incremental update
        when set to None, computation should be performed in a batch mode
    :return: List
    """
    # index all factors and vars
    ID2step_idx = {}
    max_time_step = 0
    rbt_idx = []
    lmk_idx = []
    for i in range(len(nodes)):
        if nodes[i].type == VariableType.Pose:
            rbt_idx.append(i)
            tmp_ID = nodes[i].name[0]
            tmp_step = int(nodes[i].name[1:])
            if tmp_ID not in ID2step_idx:
                ID2step_idx[tmp_ID] = {"step":[tmp_step],"var_idx":[i]}
            else:
                ID2step_idx[tmp_ID]["step"].append(tmp_step)
                ID2step_idx[tmp_ID]["var_idx"].append(i)
        else:
            lmk_idx.append(i)

    # sort var idx by steps
    for ID, step_idx in ID2step_idx.items():
        assert len(set(step_idx["step"])) == len(step_idx["step"])
        step_idx["step"],step_idx["var_idx"] = sort_pair_lists(step_idx["step"],step_idx["var_idx"])
        if max_time_step < step_idx["step"][-1]:
            max_time_step = step_idx["step"][-1]

    var2factors = {}
    for i, factor in enumerate(factors):
        if isinstance(factor, UnaryFactor):
            # prior_idx.append(i)
            var = factor.vars[0]
            var2factors = update_list_in_dict(var2factors, var, "prior", i)
        elif isinstance(factor, BinaryFactor):
            var1 = factor.var1
            var2 = factor.var2
            if var1.type == var2.type == VariableType.Pose:
                # p2p_idx.append(i)
                if isinstance(factor, SE2RelativeGaussianLikelihoodFactor):
                    if (var1.name[0] == var2.name[0]) and (ID2step_idx[var1.name[0]]["step"].index(int(var2.name[1:])) -
                                                           ID2step_idx[var1.name[0]]["step"].index(int(var1.name[1:])) == 1):
                        var2factors = update_list_in_dict(var2factors, var2, "odom", i)
                        continue
                var2factors = update_list_in_dict(var2factors, var1, "pose_obsv", i)
            elif factor.var1.type == VariableType.Pose and factor.var2.type == VariableType.Landmark:
                var2factors = update_list_in_dict(var2factors, var1, "lmk_obsv", i)
            else:
                raise ValueError("Unknown factors: ", factor.__str__())
        elif isinstance(factor, AmbiguousDataAssociationFactor):
            ob_var = factor.root_var
            if factor.child_vars[0].type == VariableType.Pose:
                var2factors = update_list_in_dict(var2factors, ob_var, "pose_obsv", i)
            elif factor.child_vars[0].type == VariableType.Landmark:
                var2factors = update_list_in_dict(var2factors, ob_var, "lmk_obsv", i)


            # ada_idx.append(i)
    # group factors and variables incrementally
    # //return values and indices of factors
    incVarFactorPairs = []
    # vector<pair<Values, vector<int>>> res;
    if (incremental_step is None or incremental_step > max_time_step+1 or incremental_step <=0):
        incremental_step = max_time_step+1
        # print("Reset incremental_step to its max feasible value: ", incremental_step)

    newVars = []
    newFactors = []
    addedRbts = set()
    addedLmks = set()

    admitted_time_steps = set()
    new_data = False
    for t_step in range(max_time_step+1):
        for id, step_idx in ID2step_idx.items():
            if t_step in step_idx["step"]:
                admitted_time_steps.add(t_step)
                new_data = True
                rbt_var_idx = step_idx["var_idx"][step_idx["step"].index(t_step)]
                tmp_var = nodes[rbt_var_idx]
                newVars.append(tmp_var)
                addedRbts.add(tmp_var)

                if tmp_var in var2factors:
                    tmp_factors = var2factors[tmp_var].values()
                    for t_f in tmp_factors:
                        newFactors += t_f

                    if "lmk_obsv" in var2factors[tmp_var]:
                        for f_idx in var2factors[tmp_var]["lmk_obsv"]:
                            lmk_vars = factors[f_idx].vars[1:]
                            lmk_diff = set(lmk_vars) - addedLmks
                            for v in lmk_diff:
                                newVars.append(v)
                                addedLmks.add(v)
                                if v in var2factors and "prior" in var2factors[v]:
                                    newFactors += var2factors[v]["prior"]
        if ((len(admitted_time_steps) % incremental_step == 0) or t_step == max_time_step) and new_data:
            n_v = [var for var in newVars]
            n_f = [factors[j] for j in newFactors]
            incVarFactorPairs.append([n_v, n_f])
            newVars = []
            newFactors = []
            new_data = False
    print("There are ", len(incVarFactorPairs), " pairs of vars and factors.")
    return incVarFactorPairs

def graph_file_parser(data_file: str, data_format: Union['fg', 'g2o', 'toro'], prior_cov_scale):
    if data_format == 'fg':
        nodes, truth, factors = read_factor_graph_from_file(data_file)
    elif data_format == 'g2o' or data_format == 'toro':
        pg = G2oToroPoseGraphReader(data_file)
        nodes, factors, truth = pg.dataForSolver(prior_cov_scale=prior_cov_scale)
    else:
        raise ValueError('Unknown data_format: ', data_format)
    return nodes, truth, factors


def incVarFactor2DRp(nodes_factors_by_step: List[Tuple[List[Variable], List[Factor]]])->np.ndarray:
    rbt_vars = []
    var2pose = {}
    odom_x = []
    odom_y = []
    for step in range(len(nodes_factors_by_step)):
        step_nodes, step_factors = nodes_factors_by_step[step]
        for f in step_factors:
            if isinstance(f, PriorFactor):
                rbt_vars.append(f.vars[0])
                var2pose[f.vars[0]] = SE2Pose(*f.observation)
                odom_y.append(var2pose[rbt_vars[-1]].y)
                odom_x.append(var2pose[rbt_vars[-1]].x)
            elif isinstance(f, SE2RelativeGaussianLikelihoodFactor):
                if f.var1 == rbt_vars[-1]:
                    var2pose[f.var2] = var2pose[f.var1] * SE2Pose(*f.observation)
                    rbt_vars.append(f.var2)
                    odom_y.append(var2pose[rbt_vars[-1]].y)
                    odom_x.append(var2pose[rbt_vars[-1]].x)
    return np.array([odom_x,odom_y])