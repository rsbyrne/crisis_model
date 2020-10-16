from everest.builts._observer import Observer
from everest.utilities import Grouper
from everest.quantity import Quantity

from ..exceptions import *

class ObserverException(CrisisModelException):
    pass
class ObserverMissingAsset(CrisisModelMissingAsset, ObserverException):
    pass

class CrisisObserver(Observer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _observer_construct(self, subject, inputs):
        observer = super()._observer_construct(subject, inputs)
        constructed = self._construct(subject.observables, inputs)
        if not 'analysers' in constructed:
            raise ObserverMissingAsset
        if not all(isinstance(an, Analyser) for an in constructed['analysers']):
            raise TypeError
        def evaluate():
            return OrderedDict(
                (k, an.evaluate())
                    for k, an in constructed['analysers'].items()
                )
        constructed['evaluate'] = evaluate
        observer.update(constructed, silent = True)
        return observer

    def _construct(self, observables, inputs):
        raise ObserverMissingAsset

class AnalyserException(ObserverException):
    pass
class AnalyserMissingAsset(CrisisModelMissingAsset, AnalyserException):
    pass

class Analyser(Quantity):
    def __init__(self):
        super().__init__()
    def evaluate(self):
        raise AnalyserMissingAsset
