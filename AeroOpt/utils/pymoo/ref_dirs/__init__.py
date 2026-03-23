
from AeroOpt.utils.pymoo.ref_dirs.reduction import ReductionBasedReferenceDirectionFactory
from AeroOpt.utils.pymoo.ref_dirs.incremental import IncrementalReferenceDirectionFactory
from AeroOpt.utils.pymoo.reference_direction import MultiLayerReferenceDirectionFactory


def get_reference_directions(name, *args, **kwargs):
    
    from AeroOpt.utils.pymoo.reference_direction import UniformReferenceDirectionFactory

    REF = {
        "uniform": UniformReferenceDirectionFactory,
        "das-dennis": UniformReferenceDirectionFactory,
        "multi-layer": MultiLayerReferenceDirectionFactory,
        "reduction": ReductionBasedReferenceDirectionFactory,
        "incremental": IncrementalReferenceDirectionFactory,
    }

    if name not in REF:
        raise Exception("Reference directions factory not found.")

    return REF[name](*args, **kwargs)()
