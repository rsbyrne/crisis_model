from collections import OrderedDict

import numpy as np

from everest import Function as Fn

from crisis_model.observers import CrisisObserver

class Epidemiology1(CrisisObserver):

    def __init__(self,
            **kwargs
            ):
        super().__init__(**kwargs)

    @staticmethod
    def _construct(o, i):

        active = Fn(Fn(o, 'indicated'), op = (np.nonzero, Fn(None, 0), len))
        recovered = Fn(Fn(o, 'recovered'), op = (np.nonzero, Fn(None, 0), len))
        cumulative = active + recovered

        analysers = OrderedDict()
        analysers['active'] = active
        analysers['recovered'] = recovered
        analysers['cumulative'] = cumulative

        return locals()
