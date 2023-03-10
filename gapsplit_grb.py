import random
import numpy as np
from time import time

import pandas as pd
from gurobipy.gurobipy import GRB
from gurobipy.gurobipy import read
from gurobipy.gurobipy import QuadExpr


def get_var_names(model):
    names = []

    for v in model.getVars():
        names.append(v.VarName)

    return names


def filter_pts_from_names(pts, names, obj_name, vars):
    inds = []
    cols = []
    obj_ind = 0

    int_inds = []
    int_cols = []

    for i, v in enumerate(vars):
        if v.VType == 'I' or v.VType == 'B':
            int_inds.append(i)
            int_cols.append(v.VarName)

    for i, n in enumerate(names):
        if 'EXC'.lower() in n.lower():
            inds.append(i)
            cols.append(n)
        elif n == obj_name:
            obj_ind = i

    samples_filtered = np.hstack((pts[:, inds], pts[:, obj_ind].reshape(-1, 1)))
    cols.append(obj_name)

    samples_filtered_int = pts[:, int_inds]

    return pd.DataFrame(samples_filtered, columns=cols, index=None), pd.DataFrame(samples_filtered_int, columns=int_cols)


def sample(fname, n_points, lower_bounds=None, upper_bounds=None, n_update=100, n_secondary=0):
    np.random.seed(int(time()))
    model = read(fname)
    model.setParam('OutputFlag', False)
    n_reactions = len(model.getVars())

    var_names = get_var_names(model)
    obj_name = model.getObjective().getVar(0).VarName

    obj_ind = 0
    obj_var = None

    # TESTING
    for i, v in enumerate(model.getVars()):
        if v.VarName == obj_name:
            obj_ind = i
            obj_var = v

    # print("Generating warmup points...")

    if not (lower_bounds and upper_bounds):
        (upper_bounds, lower_bounds) = __fva(model)

    samples = np.zeros((2, n_reactions))
    samples[1] = lower_bounds
    samples[0] = upper_bounds
    reaction_ranges = np.subtract(upper_bounds, lower_bounds)
    square_ranges = np.square(reaction_ranges)
    pt_count = 2
    unblocked = []
    cts_unblocked = []

    for rxn, rxn_range, i in zip(model.getVars(), reaction_ranges, range(n_reactions)):
        if rxn_range > 1e-9 and rxn.VType:
            unblocked.append((rxn, i))
            cts_unblocked.append((rxn, i))

        elif rxn_range > 1e-9:
            unblocked.append((rxn, i))

    ind_count = 0
    mi_flag = False

    while pt_count < n_points:

        samples = samples.T
        (reaction, ind) = unblocked[ind_count % len(unblocked)]
        target = __get_target(samples, ind, reaction.VType, lower_bounds, reaction_ranges)

        if n_secondary > 0:
            model.addLConstr(reaction, GRB.EQUAL, target, "GapSPLIT")
            objective = QuadExpr()
        elif reaction.VType == 'I' or reaction.VType == 'B':
            model.addLConstr(reaction, GRB.EQUAL, target, "GapSPLIT")
            mi_flag = True
        else:
            objective = QuadExpr((target - reaction) * (target - reaction) / square_ranges[ind])

        random_indices = random.sample(range(len(unblocked)), n_secondary)

        if mi_flag:
            random_indices = random.sample(range(len(cts_unblocked)), 1)
            mi_flag = False

        for secondary_key in random_indices:
            # TESTING
            if secondary_key != obj_ind:
                (secondary_rxn, secondary_ind) = unblocked[secondary_key]
                target = __get_target(samples, secondary_ind, secondary_rxn.VType, lower_bounds, reaction_ranges)
                objective.add((secondary_rxn - target) * (secondary_rxn - target) / square_ranges[secondary_ind])

        # TESTING
        objective.add((obj_var - upper_bounds[obj_ind]) * (obj_var - upper_bounds[obj_ind]) / square_ranges[obj_ind])

        model.setObjective(objective, GRB.MINIMIZE)
        model.update()
        model.optimize()
        samples = samples.T

        ind_count += 1
        new_pts = np.zeros(n_reactions)

        if model.status == 2:
            for i in range(n_reactions):
                new_pts[i] = model.getVars()[i].x

            pt_count += 1
            samples = np.vstack((samples, new_pts))

        if model.getConstrByName("GapSPLIT"):
            model.remove(model.getConstrByName("GapSPLIT"))

        # if n_update > 0 and (pt_count % n_update) == 0:
        #     print("\rSamples: %s\tCoverage: %s" % (pt_count, __get_coverage(samples, reaction_ranges, model.getVars())))

    return filter_pts_from_names(samples, var_names, obj_name, model.getVars())


def __get_target(points, ind, v_type, mins, ranges):
    rxn_samples = points[ind]

    if v_type == 'C':
        in_order = np.sort(rxn_samples)
        first_elements = in_order[0:len(in_order) - 1]
        last_elements = in_order[1:]

        gap_sizes = np.subtract(last_elements, first_elements)
        target_ind = np.argmax(gap_sizes)
        target = in_order[target_ind] + (in_order[target_ind + 1] - in_order[target_ind]) / 2

    else:
        values = [i for i in range(int(mins[ind]), int(mins[ind] + ranges[ind]) + 1)]
        target = values[0]
        target_freq = len(np.where(rxn_samples == values[0])[0])

        for v in values[1:]:
            if len(np.where(rxn_samples == v)[0]) < target_freq:
                target = v
                target_freq = len(np.where(rxn_samples == v)[0])

    return target


def __get_coverage(samples, ranges, rxns):
    samples = samples.T
    coverage = 0
    n = 0

    for rxn_points, rxn_range, rxn in zip(samples, ranges, rxns):

        in_order = np.sort(rxn_points)
        first_elements = in_order[0:len(in_order) - 1]
        last_elements = in_order[1:]

        gap_sizes = np.subtract(last_elements, first_elements)
        max_gap = np.amax(gap_sizes)

        if rxn_range > 1e-9 and rxn.VType == 'C':
            coverage += max_gap / rxn_range
            n += 1

    coverage = 1 - coverage / n
    return coverage


def __fva(model):
    maxes = np.zeros(len(model.getVars()))
    mins = np.zeros(len(model.getVars()))
    model.setParam('OutputFlag', False)

    for v, i in zip(model.getVars(), range(len(maxes))):
        model.setObjective(v, GRB.MAXIMIZE)
        model.update()
        model.optimize()
        maxes[i] = model.objVal

    for v, i in zip(model.getVars(), range(len(mins))):
        model.setObjective(v, GRB.MINIMIZE)
        model.update()
        model.optimize()
        mins[i] = model.objVal

    for i in range(len(maxes)):
        if maxes[i] == -0.0:
            maxes[i] = 0.0

    return maxes, mins
