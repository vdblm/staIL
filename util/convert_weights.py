"""
Imports .pk weights produced by experiments and exports
it into a format accepted by the LipSDP Lipschitz constant estimator
"""
from scipy.io import savemat, loadmat
import numpy as np
import pickle


file = open('../results/lip_expert_5_lip_policy_5/base.pk', 'rb')
data = pickle.load(file)

import pdb; pdb.set_trace()
