# Generation: one stateful Generator written against the models'
# Decodable protocol (infra/models/decodable.py). See generator.py for
# the stream/state invariant, sampling.py for the sampling knobs.

from .sampling import SamplingConfig, sample
from .generator import Generator

__all__ = ['SamplingConfig', 'sample', 'Generator']
