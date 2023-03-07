import random
import numpy as np
from time import time

from gurobipy.gurobipy import GRB
from gurobipy.gurobipy import read
from gurobipy.gurobipy import LinExpr


def get_ex_vars(model):
    ex_vars = []
    ex_var_inds = {}
    ex_var_inds_glob = []

    for i, v in enumerate(model.getVars()):
        if 'EXC'.lower() in v.VarName.lower():
            ex_vars.append(v.VarName)
            ex_var_inds_glob.append(i)

    for i, v in enumerate(ex_vars):
        ex_var_inds[v] = i

    return ex_vars, ex_var_inds, ex_var_inds_glob


def get_int_vars(model):
    ex_vars = []
    ex_var_inds = {}
    ex_var_inds_glob = []

    for i, v in enumerate(model.getVars()):
        if v.VType == 'I' or v.VType == 'B':
            ex_vars.append(v.VarName)
            ex_var_inds_glob.append(i)

    for i, v in enumerate(ex_vars):
        ex_var_inds[v] = i

    return ex_vars, ex_var_inds, ex_var_inds_glob


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


def get_bounds(model, inds):
    ubs, lbs = __fva(model)

    return np.vstack((ubs, lbs))[:, inds]


def get_target(points, ind, v_type):
    rxn_samples = points.T[ind]

    if v_type == 'C':
        in_order = np.sort(rxn_samples)
        first_elements = in_order[0:len(in_order) - 1]
        last_elements = in_order[1:]

        gap_sizes = np.subtract(last_elements, first_elements)
        target_ind = np.argmax(gap_sizes)
        target = in_order[target_ind] + (in_order[target_ind + 1] - in_order[target_ind]) / 2

    else:
        # Get least frequent value
        pass

    return target


def get_var_vals(model, vars, var_inds):
    pts = np.zeros(len(vars))

    for i, v in enumerate(vars):
        pts[i] = model.getVars()[var_inds[v]].x

    return pts


def sample(fpath):
    # Read the model from file
    model = read(fpath)
    model.setParam('OutputFlag', False)
    model_star = model.copy()

    # Get exchange var names OR integer variable names
    ex_vars, ex_var_inds, ex_var_inds_glob = get_ex_vars(model)

    int_vars, int_var_inds, int_var_inds_glob = get_int_vars(model)

    # Declare ranges for exchange vars
    bounds = get_bounds(model, ex_var_inds_glob)
    objs = []

    orig_obj = []
    input = np.zeros((len(ex_vars), len(ex_vars)))
    int_input = np.zeros((len(ex_vars), len(int_vars)))
    good_inds = []
    # While < n_samples
    for i, v in enumerate(ex_vars):
        # model_star.addLConstr(model.getVarByName(vp), GRB.EQUAL, 0, vp+'temp')
        ub = model_star.getVarByName(v).getAttr('ub')
        lb = model_star.getVarByName(v).getAttr('lb')

        model_star.getVarByName(v).setAttr('ub', 0)
        model_star.getVarByName(v).setAttr('lb', 0)
        model_star.update()

        model_star.optimize()
        model_star.update()

        if model_star.status == 2:
            objs.append(model_star.getObjective().getValue())

            for vp in ex_vars:
                var_obj = model_star.getVarByName(vp)
                input[i, ex_var_inds[vp]] = var_obj.X

            for vp in int_vars:
                var_obj = model_star.getVarByName(vp)
                int_input[i, int_var_inds[vp]] = var_obj.X

            good_inds.append(i)

        model_star.getVarByName(v).setAttr('ub', ub)
        model_star.getVarByName(v).setAttr('lb', lb)
        model_star.update()

    return input[good_inds], int_input[good_inds], np.asarray(objs)


if __name__ == '__main__':
    ec_samples, ec_objs = sample('Ecoli_core_model.lp')
    print('hi')
