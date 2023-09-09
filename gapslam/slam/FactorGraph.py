from typing import List, Set, Dict
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from slam.BayesTree import BayesTree, BayesTreeNode
from factors.Factors import Factor, UndefinedFactor, ImplicitPriorFactor
from slam.Variables import Variable, VariableType
#from external.pysuitesparse import pyccolamd


class FactorGraph(object):
    def __init__(self) -> None:
        """
        Create a FactorBayesianNetwork object
        The graph is a hybrid factor graph and Bayesian network
        """
        self._vars = []
        self._factors = []
        self._adjacent_nodes_in_factor_graph = {}
        self._adjacent_factors_from_node = {}
        self._adjacent_nodes_from_factors = {}
        self._parents_in_bayesian_network = {}

    def add_node(self, var: Variable) -> "FactorGraph":
        """
        Add a new node to the factor graph, not to the Bayesian network
        :param var:
        :return: the current entire graph
        """
        if var in self._vars:
            raise KeyError("The node has already existed in the graph")
        else:
            self._vars.append(var)
            self._adjacent_nodes_in_factor_graph[var] = set()
            self._adjacent_factors_from_node[var] = set()
        return self

    def add_factor(self, factor: Factor) -> "FactorGraph":
        """
        Add a factor to the graph
        :param factor:
        :return: the current entire graph
        """
        self._factors.append(factor)
        vars = factor.vars
        self._adjacent_nodes_from_factors[factor] = set()
        for i in range(len(vars)):
            var1 = vars[i]
            self._adjacent_factors_from_node[var1].add(factor)
            self._adjacent_nodes_from_factors[factor].add(var1)
            for j in range(i + 1, len(vars)):
                var2 = vars[j]
                self._adjacent_nodes_in_factor_graph[var1].add(var2)
                self._adjacent_nodes_in_factor_graph[var2].add(var1)
        return self

    def add_null_factor(self, vars: List[Variable]) -> "FactorGraph":
        """
        Add an undefined factor to the graph
        :param vars: nodes to be connected
        :return: the current entire graph
        """
        self.add_factor(UndefinedFactor(vars=vars))
        return self

    @property
    def factors(self) -> List[Factor]:
        return self._factors

    def eliminate_from_factor_graph_for_analysis(self, var: Variable) \
            -> "FactorGraph":
        """
        Eliminate a node from the factor graph, and add it to the Bayesian
            network
        The method is only to analyzing the chordal graph structure,
            and does not really compute the transport maps
        Dictionaries for factors are not updated
        :param var: the node to eliminate from the factor graph
        :return: the current entire graph
        """
        if var in self._parents_in_bayesian_network:
            raise KeyError("The node has already existed in the Bayesian "
                           "network")
        else:
            separator = deepcopy(self.get_neighbors_in_factor_graph(var))
            for neighbor in separator:
                self._adjacent_nodes_in_factor_graph[neighbor].remove(var)
                self._adjacent_nodes_in_factor_graph[var].remove(neighbor)
            if separator:
                self.add_null_factor(list(separator))
            self._parents_in_bayesian_network[var] = separator
        return self

    def get_neighbors_in_factor_graph(self, key: Variable) -> Set[Variable]:
        return self._adjacent_nodes_in_factor_graph[key]

    def get_adjacent_factors_from_node(self, key: Variable) -> Set[Factor]:
        return self._adjacent_factors_from_node[key]

    def get_adjacent_nodes_from_factor(self, factor: Factor) -> Set[Variable]:
        return self._adjacent_nodes_from_factors[factor]

    def get_parents_in_bayesian_network(self, key: Variable) -> Set[Variable]:
        return self._parents_in_bayesian_network[key]

    def analyze_elimination_ordering(self, method: str = "ccolamd", last_vars: List[Variable] = None) -> List[Variable]:
        """
        use cholmod to anlyze the elimination order of the keys except
        last_keys and then put last_keys to the end of the analyzed ordering
        :param method: the method to generate ordering
            supported methods are:
                1. natural      ordering by which variables are added
                2. ccolamd      ordering by constrained colamd algorithm
        :param last_vars: nodes to be put at the end when using ccolamd
        :return: the elimination ordering
        """
        if method == "natural":
            ordering = sorted(self._vars)
        elif method == "ccolamd":
            if not last_vars:
                last_vars = [[var for var in self._vars if var.type ==
                              VariableType.Pose][-1]]
            num_vars = len(self._vars)
            num_factors = len(self._factors)
            var_to_ind = {var: index for index, var in enumerate(self._vars)}
            rows = []
            cols = []
            data_mapping = {}
            for i, factor in enumerate(self._factors):
                for var in factor.vars:
                    j = var_to_ind[var]
                    rows.append(i)
                    cols.append(j)
                    if (i, j) in data_mapping:
                        data_mapping[(i, j)] += 1
                    else:
                        data_mapping[(i, j)] = 1
            rows = np.array(rows)
            cols = np.array(cols)
            data = [data_mapping[rows[i], cols[i]] for i in range(len(rows))]
            incidence = sp.coo_matrix((data, (rows, cols)), shape=(num_factors,
                                                                   num_vars))
            cmember = [0 for _ in range(num_vars)]
            for var in last_vars:
                cmember[var_to_ind[var]] = 1
            if all(c == 1 for c in cmember):
                cmember = [0] * len(cmember)
            incidence = incidence.tocsc()
            ordered_indices = pyccolamd(S=incidence, cmember=cmember)
            ordering = [self._vars[index] for index in ordered_indices]
        else:
            raise ValueError("Unrecognized method for analyzing "
                             "elimination order")
        return ordering

    def convert_to_bayesian_network_for_analysis(
            self, ordering: List[Variable]) -> "FactorGraph":
        """
        With specified ordering, get the resulted Bayesian network without
        really computing transport maps
        :param ordering: the elimination ordering
        :return: the converted graph
        """
        for var in ordering:
            self.eliminate_from_factor_graph_for_analysis(var)
        return self

    @property
    def vars(self) -> List[Variable]:
        return self._vars

    def get_bayes_tree(self, ordering: List[Variable] = None,
                       method: str = "ccolamd",
                       last_vars: List[Variable] = None) -> BayesTree:
        """
        Get the Bayes tree with specified ordering
        If no ordering is specified, then an ascending ordering is used
        :param ordering:
        :param method: when no ordering is given, used to generate ordering
        :param last_vars: when no ordering is given, used to generate ordering
        :return: the constructed Bayes tree
        """
        if ordering is None:
            ordering = self.analyze_elimination_ordering(method=method,
                                                         last_vars=last_vars)
        # TODO: delete copy
        copy = FactorGraph()
        copy._vars = deepcopy(self._vars)
        copy._parents_in_bayesian_network = deepcopy(
            self._parents_in_bayesian_network)
        copy._adjacent_nodes_in_factor_graph = deepcopy(
            self._adjacent_nodes_in_factor_graph.copy())
        copy._adjacent_factors_from_node = {var: set() for var
                                            in copy._vars}
        copy.convert_to_bayesian_network_for_analysis(ordering)
        bayes_tree = BayesTree(frontal=ordering[-1])
        bayes_tree.reverse_elimination_order = ordering[::-1]
        for frontal in ordering[:-1][::-1]:
            bayes_tree.add_node(
                frontal=frontal,
                parents=copy.get_parents_in_bayesian_network(frontal))
        return bayes_tree

    def get_sub_factor_graph_with_prior(self, variables: Set[Variable],
                                        sub_trees: List[BayesTree],
                                        clique_prior_dict: Dict[BayesTreeNode, ImplicitPriorFactor]) \
            -> "FactorGraph":
        """
        Get a sub factor graph containing all specified variables
        """
        subgraph = FactorGraph()
        for node in self._vars:
            if node in variables:
                subgraph.add_node(node)

        for factor in self._factors:
            if set(factor.vars).issubset(variables):
                not_in_subtree = True
                for tree in sub_trees:
                    if set(factor.vars).issubset(tree.root.vars):
                        not_in_subtree = False
                        break
                if not_in_subtree:
                    subgraph.add_factor(factor)
        for subtree in sub_trees:  # factors from previous elimination
            subgraph.add_factor(
                clique_prior_dict[subtree.root])
        return subgraph

    def eliminate_clique_variables(self, clique: BayesTreeNode,
                                         new_factor: ImplicitPriorFactor) \
            -> "FactorGraph":
        """
        Eliminate clique variables and factors and append a new factor
        """
        subgraph = FactorGraph()
        for node in self._vars:
            if node not in clique.frontal:
                subgraph.add_node(node)

        for factor in self._factors:
            if not (set(factor.vars).issubset(clique.vars)):
                subgraph.add_factor(factor)
        if new_factor is not None:
            subgraph.add_factor(
                new_factor)
        return subgraph

    def get_clique_factor_graph(self, clique: BayesTreeNode) \
            -> "FactorGraph":
        """
        Get a sub factor graph on the clique
        """
        subgraph = FactorGraph()
        for node in self._vars:
            if node in clique.vars:
                subgraph.add_node(node)

        for factor in self._factors:
            if set(factor.vars).issubset(clique.vars):
                subgraph.add_factor(factor)

        return subgraph

    @staticmethod
    def generate_pose_first_ordering(nodes) -> None:
        """
        Generate the ordering by which nodes are added and lmk eliminated later
        """
        pose_list = []
        lmk_list = []
        for node in nodes:
            if node._type == VariableType.Landmark:
                lmk_list.append(node)
            else:
                pose_list.append(node)
        return pose_list + lmk_list