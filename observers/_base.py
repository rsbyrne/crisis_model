from collections import OrderedDict

from everest.builts._observer import Observer, ObserverConstruct
from everest.utilities import Grouper

from ..exceptions import *

class CrisisObserver(Observer):

    def __init__(self, keys, **kwargs):
        self._keys = lambda: keys
        super().__init__(**kwargs)

    def _construct(self, subject):
        analysers = self._user_construct(subject)
        analysers = dict(zip(self.keys(), analysers))
        return ObserverConstruct(self, subject, **analysers)

    def _user_construct(self, observables, inputs):
        raise MissingAsset("User must provide _user_construct method.")
