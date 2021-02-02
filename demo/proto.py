# %%
import os, sys
import numpy as np
import time

# Add path to package directory to access main module using absolute import
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '..')))

from kcsp import utils

basedir = '/home/cmendoza/Research/sikmeans/LKini2019/data/Study012'
dirpath = os.path.join(basedir, 'interictal')
fpath1 = os.path.join(dirpath, 'monte7_data_split_for_trainTest.mat')
i_start = utils.loadmat73(fpath1, 'i_start')
i_start = utils.apply2list(i_start, np.squeeze)
fun = lambda x : x - 1 # Matlab index starts at 1, Python at 0
i_start = utils.apply2list(i_start, fun)
dfname = utils.loadmat73(fpath1, 'fnames')

wdir = '/home/cmendoza/Research/sikmeans/LKini2019/results/NER21/HUP078'
wfname = 'results_dmonte7_trainTest_band5_regular.mat'
wpath = os.path.join(wdir, wfname)
W = utils.loadmat73(wpath, 'W')
W = W[:79,:]
os.environ['SLURM_NTASKS_PER_NODE'] = '4'

# W = np.random.rand(79,2)
# %% Extract training data from pre-ictal and apply CSP filter
i_set = 0 # 0: training, 1: testing
winlen = 512
i_start = [i_start[ii][i_set] for ii in range(len(i_start))]
tic = time.perf_counter()
X = utils.getCSPdata(dirpath, dfname, i_start, winlen, W)
toc = time.perf_counter()
print(f"Got CSP data in {toc - tic:0.4f} seconds")

# %%
