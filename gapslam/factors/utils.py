from typing import List, Tuple

from factors.Factors import PriorFactor, BinaryFactor, KWayFactor, AmbiguousDataAssociationFactor, \
    BinaryFactorWithNullHypo
import numpy as np
import gtsam


def classify_factors(factors: List, ranked_classes: List):
    factor_groups = [[] for _ in range(len(ranked_classes))]
    for factor in factors:
        classified = False
        for i, factor_class in enumerate(ranked_classes):
            if isinstance(factor, factor_class):
                factor_groups[i].append(factor)
                classified = True
                break
        if not classified:
            raise ValueError("Unknown factor classes: "+factor.__str__())
    return factor_groups


def unpack_prior_binary_nh_da_factors(factors: List)->\
        Tuple[List[PriorFactor], List[BinaryFactor], List[BinaryFactorWithNullHypo], List[AmbiguousDataAssociationFactor]]:
    pr, null_hypo, da, bf = classify_factors(factors, [PriorFactor,
                                                       BinaryFactorWithNullHypo,
                                                       AmbiguousDataAssociationFactor,
                                                       BinaryFactor])
    return pr, bf, null_hypo, da