import numpy as np
from scipy import spatial

def get_coordInfo(corner, aspect, scale):
    minCoords = np.array(corner)
    maxCoords = np.array([aspect, 1.]) * scale
    domainLengths = maxCoords - minCoords
    return minCoords, maxCoords, domainLengths

def reshape_coords(coords, mins, maxs, lengths):
    coords = coords.copy()
    oldMins, newMins = mins
    oldMaxs, newMaxs = maxs
    oldLengths, newLengths = lengths
    coords[...] = coords - oldMins
    coords[...] = coords / oldLengths * newLengths
    coords[...] = coords + newMins
    return coords

def wrap_coords(coordArr, minCoords, maxCoords):
    minCoords, maxCoords = np.array(minCoords), np.array(maxCoords)
    domainLengths = maxCoords - minCoords
    coordArr[...] = (coordArr - minCoords) / domainLengths # is now unit
    coordArr[...] = coordArr % 1
    coordArr[...] = (coordArr * domainLengths) + minCoords
#     assert np.all(minCoords <= coordArr <= maxCoords)

def displace_coords(coordArr, lengths, headings, wrap = None):
    displacements = np.stack((
        np.cos(headings),
        np.sin(headings),
        ), axis = -1
        ) * lengths[:, None]
    coordArr[...] = coordArr + displacements
    if not wrap is None:
        minCoords, maxCoords = wrap
        wrap_coords(coordArr, minCoords, maxCoords)

def random_displace_coords(coordArr, length, rng, **kwargs):
    lengths = rng.random(len(coordArr)) * length
    headings = rng.random(len(coordArr)) * 2. * np.pi
    displace_coords(coordArr, lengths, headings, **kwargs)

def round_coords(coordArr, spatialDecimals):
    coordArr[...] = np.round(coordArr, spatialDecimals)
    if spatialDecimals == 0:
        coordArr[...] = coordArr.astype(int)

def resize_arr(arr, indices, subtract = False):
    if subtract:
        return np.delete(arr, indices, axis = 0)
    else:
        return np.append(arr, arr[indices])

def swarm_split(
        arr,
        corners,
        aspects,
        scales,
        popDensity,
        spatialDecimals = None,
        ):
    oldInfo, newInfo = (
        get_coordInfo(*data)
            for data in zip(corners, aspects, scales)
        )
    newPop = int(aspects[1] * scales[1] ** 2 * popDensity)
    oldPop = len(arr)
    addPop = newPop - oldPop
    subtract = addPop < 0
    rng = np.random.default_rng(oldPop * newPop)
    indices = rng.choice(np.arange(oldPop), abs(addPop))
    if len(arr.shape) == 1 or subtract:
        return resize_arr(arr, indices, subtract)
    else:
        new = arr[indices]
        dispLength = np.mean(spatial.distance.pdist(arr))
        random_displace_coords(new, dispLength, rng, wrap = newInfo[:2])
        arr = np.vstack([arr, new])
        if not spatialDecimals is None:
            round_coords(arr, spatialDecimals)
        return arr

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
