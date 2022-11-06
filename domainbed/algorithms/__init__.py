from .algorithms import *


ALGORITHMS = [
        'ERM',
        'Fish',
        'IRM',
        'GroupDRO',
        'Mixup',
        'MLDG',
        'CORAL',
        'MMD',
        'DANN',
        'CDANN',
        'MTL',
        'SagNet',
        'ARM',
        'VREx',
        'RSC',
        'SD',
        'ANDMask',
        'SANDMask',
        'IGA',
        'SelfReg',
        "Fishr",
        'TRM',
        'IB_ERM',
        'IB_IRM',
        'CAD',
        'CondCAD',
        'Transfer',
        'CausIRL_CORAL',
        'CausIRL_MMD',

]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
