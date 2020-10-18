from collections import OrderedDict

from everest.builts._observer import Observer
from everest.utilities import Grouper

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
