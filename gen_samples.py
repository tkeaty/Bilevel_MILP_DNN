import gapsplit_grb
import numpy as np
from gurobipy.gurobipy import read

# ec_pts = gapsplit_grb.sample('Ecoli_core_model.lp', 1000)
# ec_pts.to_csv('Ecoli_1k_exch.csv')

# ind_pts = gapsplit_grb.sample('ind750.lp', 10000)
# np.savetxt('ind750_10k_exch.csv', ind_pts, delimiter=',')

# model = read('iSMU_srfba.mps')
# for v in model.getVars():
#     if v.VType != 'C':
#         print(v.VarName)


#
# ind_pts = None
#
# for i in range(25):
#     if i == 0:
#         ind_pts = gapsplit_grb.sample('ind750.lp', 4000)
#     else:
#         new_pts = gapsplit_grb.sample('ind750.lp', 4000)
#         ind_pts = np.vstack((ind_pts, new_pts))
#
# np.savetxt('ind750_10k.csv', ind_pts, delimiter=',')
#
# # check why this is glitchy
pao_pts = gapsplit_grb.sample('pao.lp', 10000, n_secondary=10)
np.savetxt('pao_10k_exch.csv', pao_pts, delimiter=',')

# ec_model = read('Ecoli_core_model.lp')
# ind_model = read('ind750.lp')
# pao_model = read('pao.lp')
#
# print('Ecoli:')
# print('Constraints: %i' % len(ec_model.getConstrs()))
# print('Vars: %i' % len(ec_model.getVars()))
#
# print('ind750:')
# print('Constraints: %i' % len(ind_model.getConstrs()))
# print('Vars: %i' % len(ind_model.getVars()))
#
# print('Pseudomonas:')
# print('Constraints: %i' % len(pao_model.getConstrs()))
# print('Vars: %i' % len(pao_model.getVars()))

# import torch
# import math
# # this ensures that the current MacOS version is at least 12.3+
# print(torch.backends.mps.is_available())
# # this ensures that the current current PyTorch installation was built with MPS activated.
# print(torch.backends.mps.is_built())
