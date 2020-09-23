import numpy as np
from functools import partial
from scipy import spatial

from everest.builts._iterator import Iterator
from everest.value import Value
from everest.utilities import Grouper

from window.plot import Canvas, Data

colourCodes = dict(zip(
    ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'pink', 'grey'],
    [1. / 18. + i * 1. / 9 for i in range(0, 9)],
    ))

class EndModel(Exception):
    pass

class Model001(Iterator):

    def __init__(self,
            aspect = 1.2,
            scale = 22.,
            corner = [0., 0.],
            popDensity = 508,
            initialIndicated = 1,
            directionChange = 0.5,
            travelLength = 0.5,
            infectionChance = 0.1,
            recoverChance = 0.1,
            contactLength = 0.01,
            spatialDecimals = None,
            seed = 1066,
            ):

        nAgents = int(scale ** 2 * aspect * popDensity)

        rng = np.random.default_rng(seed)
        rngInitState = rng.__getstate__()
        rngCount = Value(0)
        def random(*args, **kwargs):
            rngCount.value += 1
            return rng.random(*args, **kwargs)
        def choice(*args, **kwargs):
            rngCount.value += 1
            return rng.choice(*args, **kwargs)

        minCoords = np.array(corner)
        maxCoords = np.array([aspect, 1.]) * scale
        domainLengths = maxCoords - minCoords

        allIndices = np.arange(nAgents)

        agentCoords = np.empty(shape = (nAgents, 2), dtype = float)

        headings = np.empty(shape = nAgents, dtype = float)
        distances = np.empty(shape = nAgents, dtype = float)

        indicated = np.empty(nAgents, dtype = bool)
        recovered = np.empty(nAgents, dtype = bool)
        susceptible = np.empty(nAgents, dtype = bool)

        def update_coords():
            distances[...] = random(nAgents) * travelLength
            headings[...] = headings + (random(nAgents) - 0.5) * 2. * np.pi * directionChange
            displacements = np.stack([
                np.cos(headings),
                np.sin(headings)
                ], axis = -1
                ) * distances[:, None]
            agentCoords[...] = agentCoords + displacements % domainLengths
            agentCoords[...] = np.where(
                agentCoords < minCoords,
                maxCoords - agentCoords,
                agentCoords
                )
            agentCoords[...] = np.where(
                agentCoords > maxCoords,
                minCoords + agentCoords - maxCoords,
                agentCoords
                )
            if not spatialDecimals is None:
                agentCoords[...] = np.round(agentCoords, spatialDecimals)
                if spatialDecimals == 0:
                    agentCoords[...] = agentCoords.astype(int)

        def get_encounters():
            susceptibles = susceptible.nonzero()[0]
            indicateds = indicated.nonzero()[0]
            if not (len(susceptibles) and len(indicateds)):
                return []
            flip = len(susceptibles) < len(indicateds)
            if not flip:
                ids1, ids2 = susceptibles, indicateds
            else:
                ids1, ids2 = indicateds, susceptibles
            adjCoords = agentCoords - minCoords
            strategy = partial(accelerated_neighbours_radius_array, leafsize = 128)
            contacts = strategy(
                adjCoords[ids1],
                adjCoords[ids2],
                contactLength,
                maxCoords - minCoords,
                )
            encounters = np.array([
                (id1, id2)
                    for id1, subcontacts in zip(ids2, contacts) 
                        for id2 in ids1[subcontacts]
                ]) # <- very quick
            if len(encounters) and flip:
                encounters = encounters[:, slice(None, None, -1)]
            return encounters

        def initialise():
            rng.__setstate__(rngInitState)
            agentCoords[...] = random((nAgents, 2)) * (maxCoords - minCoords) + corner
            headings[...] = random(nAgents) * 2. * np.pi
            indicated[...] = False
            indicated[choice(allIndices, initialIndicated)] = True
            recovered[...] = False
            susceptible[...] = ~indicated

        def iterate():
            if not len(indicated.nonzero()[0]):
                raise EndModel
            update_coords()
            encounters = get_encounters()
            if len(encounters):
                newIndicateds = np.unique(
                    encounters[random(encounters.shape[0])
                        < infectionChance][:, 1]
                    )
                indicated[newIndicateds] = True
            else:
                newIndicateds = []
            indicateds = indicated.nonzero()[0]
            recovery = random(len(indicateds)) < recoverChance
            indicated[indicateds] = ~recovery
            recovered[indicateds] = recovery
            susceptible[newIndicateds] = False

        def out():
            return [agentCoords.copy(), indicated.copy(), recovered.copy(), rngCount.value]

        outkeys = ['agentCoords', 'indicated', 'recovered', 'rngCount']

        def load(loadDict):
            rng.__setstate__(rngInitState)
            rngCount.value = loadDict['rngCount']
            ignoreMe = [rng.random() for i in range(rngCount.value)]
            agentCoords[...] = loadDict['agentCoords']
            indicated[...] = loadDict['indicated']

        self._initialise = initialise
        self._iterate = iterate
        self._out = out
        self._outkeys = outkeys
        self._load = load

        super().__init__()

        self.step = self.count
        self.locals = Grouper(locals())

    def show(self):

        global colourCodes

        aspect = self.locals.aspect
        minCoords = self.locals.minCoords
        maxCoords = self.locals.maxCoords
        coords = self.locals.agentCoords
        indicated = self.locals.indicated
        recovered = self.locals.recovered
        susceptible = self.locals.susceptible
        nMarkers = self.locals.nAgents
        step = self.step

        if not hasattr(self, 'fig'):

            figScale = int(np.log10(nMarkers)) + 3

            xs, ys = coords.transpose()

            cs = np.random.rand(nMarkers)

            figsize = (int(figScale * aspect), figScale)
            canvas = Canvas(size = figsize)
            ax = canvas.make_ax()
            ax.scatter(
                Data(
                    xs,
                    lims = (minCoords[0], maxCoords[0]),
                    capped = (True, True),
                    label = 'x km',
                    ),
                Data(
                    ys,
                    lims = (minCoords[1], maxCoords[1]),
                    capped = (True, True),
                    label = 'y km'
                    ),
                cs,
                )
            ax.ax.set_facecolor('black')

#             ax.toggle_tickLabels_x()
#             ax.toggle_tickLabels_y()
#             ax.toggle_label_x()
#             ax.toggle_label_y()

            collection = ax.collections[0]
            collection.set_alpha(1.)
            collection.set_cmap('Set1')
            collection.autoscale()

            self.fig = canvas

        figScale = self.fig.size[1]

        cs = np.zeros(nMarkers)
        cs[...] = colourCodes['grey']
        cs[susceptible] = colourCodes['blue']
        cs[indicated] = colourCodes['red']
        cs[recovered] = colourCodes['yellow']

        nSqPoints = figScale ** 2 * aspect * 72 ** 2
        s = nSqPoints / nMarkers * 0.1
        ss = np.full(nMarkers, s)
        ss[indicated] *= 4

        canvas = self.fig
        ax = canvas.axes[0][0][0]

        collection = ax.collections[0]
        collection.set_offsets(np.concatenate([coords[~indicated], coords[indicated]]))
        collection.set_array(np.concatenate([cs[~indicated], cs[indicated]]))
        collection.set_sizes(np.concatenate([ss[~indicated], ss[indicated]]))

        ax.set_title(f'Step: {str(step)}')

        return canvas.fig

def accelerated_neighbours_radius_array(
        coords,
        targets,
        radius,
        domainLengths,
        leafsize = 128,
        ):
    kdtree = spatial.cKDTree(
        coords,
        compact_nodes = True,
        balanced_tree = True,
        leafsize = leafsize,
        boxsize = domainLengths + 1e-9,
        )
    contacts = kdtree.query_ball_point(targets, radius)
    return [np.array(row, dtype = int) for row in contacts]

CLASS = Model001