from functools import wraps
import numpy as np
from collections import OrderedDict

from window.plot import Canvas, Data

from everest.builts._wanderer import Wanderer
from everest.builts._stateful import Statelet
from everest.builts._chroner import Chroner
from everest.builts._observable import Observable, _observation_mode
from everest.utilities import Grouper
from everest.builts._voyager import _voyager_initialise_if_necessary

from ..exceptions import *

class SystemException(CrisisModelException):
    pass
class SystemMissingAsset(CrisisModelMissingAsset, SystemException):
    pass

from ..array import swarm_split

colourCodes = dict(zip(
    ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'pink', 'grey'],
    [1. / 18. + i * 1. / 9 for i in range(0, 9)],
    ))

class SystemVar(Statelet):
    def __init__(self, var, name, params):
        super().__init__(var, name)
        self.params = params
    def _imitate(self, fromVar):
        self.data[...] = swarm_split(
            fromVar.data,
            (fromVar.params.corner, self.params.corner),
            (fromVar.params.aspect, self.params.aspect),
            (fromVar.params.scale, self.params.scale),
            self.params.popDensity,
            spatialDecimals = self.params.spatialDecimals
            )

def _constructed(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'locals'):
            self.construct()
        return func(self, *args, **kwargs)
    return wrapper

class System(Observable, Chroner, Wanderer):

    reqAtts = {
        'initialise',
        'iterate',
        '_update',
        'nAgents',
        'minCoords',
        'maxCoords',
        'susceptible',
        }
    configsKeys = (
        'agentCoords',
        'indicated',
        'recovered',
        'timeIndicated',
        )
    reqAtts.update(configsKeys)

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
            )
        self._figUpToDate = False

    def construct(self):
        localObj = Grouper(self._construct(p = self.inputs))
        del localObj['p']
        self._construct_check(localObj)
        self.locals = localObj
        self._stateVars = [
            SystemVar(self.locals[k], k, self.inputs)
                for k in self.configs.keys()
            ]
        self.observables.clear()
        self.observables.update(self.locals)
        self._fig = self._make_fig()
    @classmethod
    def _construct_check(cls, obj):
        if not all(hasattr(obj, att) for att in cls.reqAtts):
            raise SystemMissingAsset

    @_constructed
    def _state_vars(self):
        for o in super()._state_vars(): yield o
        for v in self._stateVars: yield v

    @_constructed
    def _initialise(self):
        super()._initialise()
        self.locals.initialise()
    @_voyager_initialise_if_necessary(post = True)
    def _iterate(self):
        self.locals.iterate()
        super()._iterate()
    # @_constructed
    #
    #     outs.update(self.evaluate())
    #     return outs
    # def _evaluate(self):
    #     return OrderedDict((k, v.data.copy()) for k, v in self.state.items())

    def _save(self):
        super()._save()
        self.update()

    def update(self):
        self.locals._update()
        self._figUpToDate = False

    def _voyager_changed_state_hook(self):
        super()._voyager_changed_state_hook()
        self.update()

    def _make_fig(self):
        nMarkers = self.locals.nAgents
        figScale = int(np.log10(nMarkers)) + 3
        xs, ys = self.locals.agentCoords.transpose()
        cs = np.random.rand(nMarkers)
        figsize = (int(figScale * self.inputs.aspect), figScale)
        canvas = Canvas(size = figsize)
        ax = canvas.make_ax()
        ax.scatter(
            Data(
                xs,
                lims = (self.locals.minCoords[0], self.locals.maxCoords[0]),
                capped = (True, True),
                label = 'x km',
                ),
            Data(
                ys,
                lims = (self.locals.minCoords[1], self.locals.maxCoords[1]),
                capped = (True, True),
                label = 'y km'
                ),
            cs,
            )
        ax.ax.set_facecolor('black')
        collection = ax.collections[0]
        collection.set_alpha(1.)
        collection.set_cmap('Set1')
        collection.autoscale()
        return canvas
    @property
    @_voyager_initialise_if_necessary(post = True)
    def fig(self):
        if not self._figUpToDate:
            self._update_fig()
        return self._fig
    def _update_fig(self):
        global colourCodes
        step = self.indices.count
        indicated = self.locals.indicated
        susceptible = self.locals.susceptible
        recovered = self.locals.recovered
        coords = self.locals.agentCoords
        nMarkers = self.locals.nAgents
        figScale = self._fig.size[1]
        cs = np.zeros(nMarkers)
        cs[...] = colourCodes['grey']
        cs[susceptible] = colourCodes['blue']
        cs[indicated] = colourCodes['red']
        cs[recovered] = colourCodes['yellow']
        nSqPoints = figScale ** 2 * self.inputs.aspect * 72 ** 2
        s = nSqPoints / nMarkers * 0.1
        ss = np.full(nMarkers, s)
        ss[indicated] *= 4
        canvas = self._fig
        ax = canvas.axes[0][0][0]
        collection = ax.collections[0]
        collection.set_offsets(
            np.concatenate([coords[~indicated], coords[indicated]])
            )
        collection.set_array(
            np.concatenate([cs[~indicated], cs[indicated]])
            )
        collection.set_sizes(
            np.concatenate([ss[~indicated], ss[indicated]])
            )
        ax.set_title(f'Step: {str(step)}')
        self._figUpToDate = True
    def show(self):
        return self.fig.fig

    @property
    def count(self):
        return self.indices.count
    @property
    def chron(self):
        return self.indices.chron
