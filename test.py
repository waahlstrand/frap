"""A small test for Bayesian optimization, using hyperopt
"""

from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL
import numpy as np

space = {"n_hidden" : hp.randint('n_hidden', 16, 64),
         "lr": hp.loguniform("lr", 0.1, 0.001)}

def tune(space):

    