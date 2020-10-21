import numpy as np
from functools import partial
from scipy import spatial
from collections import OrderedDict
import warnings

from crisis_model.systems import System
from crisis_model.array import *
from crisis_model.observers import Epidemiology1

class EndModel(Exception):
    pass

class Covid1(System):

    def __init__(self,
            aspect = 1.2, # x length relative to y length
            scale = 22., # y length in km
            corner = [0., 0.], # coords of bottom-left corner
            timescale = 1., # days per timestep
            popDensity = 508, # people per sq km
            initialIndicated = 1, # initial mystery cases
            directionChange = 0.5, # where 1 == 180 deg per day
            speed = 5, # agent travel speed in km / h
            infectionChance = 0.1, # chance of transmission by 'contact'
            recoverMean = 14, # average recovery time in days
            recoverSpread = 2, # standard deviations of recovery curve
            contactLength = 1.5, # proximity in metres defining 'contact'
            spatialDecimals = None, # spatial precision limit
            seed = 1066, # random seed
            # CONFIGS (_ghost_)
            agentCoords = None,
            indicated = False,
            recovered = False,
            timeIndicated = 0.,
            ):

        super().__init__()

        self.observerClasses.append(Epidemiology1)

    def _construct(self, p):

        nAgents = int(p.scale ** 2 * p.aspect * p.popDensity)
        travelLength = p.speed * p.timescale * 24.

        minCoords, maxCoords, domainLengths = get_coordInfo(
            p.corner, p.aspect, p.scale
            )

        agentCoords = np.empty(shape = (nAgents, 2), dtype = float)

        headings = np.empty(shape = nAgents, dtype = float)
        distances = np.empty(shape = nAgents, dtype = float)

        indicated = np.empty(nAgents, dtype = bool)
        timeIndicated = np.zeros(nAgents, dtype = float)
        recovered = np.empty(nAgents, dtype = bool)
        susceptible = np.empty(nAgents, dtype = bool)

        def update_coords():
            rng = self.locals.rng
            distances[...] = rng.random(nAgents) * travelLength
            ang = p.directionChange * p.timescale
            headings[...] += (rng.random(nAgents) - 0.5) * 2. * np.pi * ang
            wrap = minCoords, maxCoords
            displace_coords(agentCoords, distances, headings, wrap)
            if not p.spatialDecimals is None:
                round_coords(agentCoords, p.spatialDecimals)

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
            strategy = partial(
                accelerated_neighbours_radius_array,
                leafsize = 128
                )
            contacts = strategy(
                adjCoords[ids1],
                adjCoords[ids2],
                p.contactLength / 1000.,
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

        def get_stepSeed():
            stepSeed = int(
                np.random.default_rng(
                    int(p.seed + self.count.value)
                    ).integers(0, int(1e9))
                )
            # print(seed, self.count.value, stepSeed, type(stepSeed))
            return stepSeed

        def initialise():
            rng = self.locals.rng = np.random.default_rng(get_stepSeed())
            if not np.all(agentCoords < np.inf):
                warnings.warn(
                    "Setting agent coords randomly - did you expect this?"
                    )
                agentCoords[...] = \
                    rng.random((nAgents, 2)) \
                    * (maxCoords - minCoords) \
                    + p.corner
            headings[...] = rng.random(nAgents) * 2. * np.pi
            susceptible[...] = True
            susceptible[indicated] = False
            susceptible[recovered] = False
            nonSusceptible = susceptible.nonzero()[0]
            nNew = min(len(nonSusceptible), p.initialIndicated)
            newCases = rng.choice(nonSusceptible, nNew, replace = False)
            indicated[newCases] = True
            susceptible[newCases] = False

        def iterate():
            rng = self.locals.rng = np.random.default_rng(get_stepSeed())
            if not len(indicated.nonzero()[0]):
                raise EndModel
            update_coords()
            encounters = get_encounters()
            if len(encounters):
                newIndicateds = np.unique(
                    encounters[
                        rng.random(encounters.shape[0]) < p.infectionChance
                        ][:, 1]
                    )
                indicated[newIndicateds] = True
            else:
                newIndicateds = []
            indicateds = indicated.nonzero()[0]
            recovery = rng.normal(
                p.recoverMean,
                p.recoverSpread,
                len(indicateds),
                ) < timeIndicated[indicated]
            indicated[indicateds] = ~recovery
            recovered[indicateds] = recovery
            susceptible[newIndicateds] = False
            timeIndicated[indicated] += p.timescale
            self.indices.chron.value += p.timescale

        def _update():
            susceptible[...] = True
            susceptible[indicated] = False
            susceptible[recovered] = False

        ret = locals()
        del ret['self']
        return ret

CLASS = Covid1
