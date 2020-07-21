import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.approaches.acfa.dual import DualObjSampler
from src.approaches.acfa.prob import ProbSampler
from src.approaches.apcs.acc_improvement import AccImprovementSampler
from src.approaches.apcs.pal_acs import PAL_ACS_Sampler
from src.approaches.apcs.redistricting import RedistrictingSampler
from src.approaches.em.sampling import EMStochasticSampler
from src.baselines.bin_pal_acs import BinaryPAL_ACS_Sampler
from src.baselines.bin_redistricting import BinaryRedistrictingSampler
from src.baselines.perfect_info import PIUSampler
from src.baselines.random import RandomSampler
from src.experiment import prepare_dataset, run_experiment
from src.utils.synth import get_synthetic_dataset


def run():
    """Runs the active classification feature selection experiment."""
    print("Running acfs.")
    print("Starting experiment.")

    sf, cf, y = get_synthetic_dataset(correlation="indirect", size=30)
    run_experiment(
        sf, cf, y,
    )

    print("Finished experiment.")
