from everest.functions import Fn
from everest.functions.misc import nonzero

from crisis_model.observers import CrisisObserver

class Epidemiology1(CrisisObserver):

    def __init__(self,
            **kwargs
            ):
        super().__init__(
            ['active', 'recovered', 'cumulative'],
            **kwargs
            )

    def _user_construct(self, subject):

        get_data = Fn(subject).get('state', Fn(), 'data')
        headcount = nonzero.close(get_data)

        active = headcount.close('indicated')
        recovered = headcount.close('recovered')
        cumulative = active + recovered

        return active, recovered, cumulative
