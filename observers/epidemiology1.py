from collections import OrderedDict

import numpy as np

from crisis_model.observers import CrisisObserver, Analyser

class Epidemiology1(CrisisObserver):

    def __init__(self,
            **kwargs
            ):
        super().__init__(**kwargs)

    @staticmethod
    def _construct(s, i):

        active = NonZero(s.indicated)
        recovered = NonZero(s.recovered)
        cumulative =

        return locals()

class NonZero(Analyser):
    def __init__(self, arr, inv = False):
        self.arr = arr
        self.inv = inv
        super().__init__()
    def evaluate(self):
        out = len(self.arr.nonzero())
        if self.inv:
            out = not out
        return out

        susceptible = s.susceptible
        indicated = s.indicated
        recovered = s.recovered
        population = observables.nAgents
        agentIDs = np.arange(population)
