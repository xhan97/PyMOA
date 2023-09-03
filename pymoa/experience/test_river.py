from river import cluster
from river import stream
from river.datasets import Insects

import os
import sys

# temporary solution for relative imports in case pyod is not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print(sys.path)


from metrics.metrics import F1_score_P, F1_score_R, purity_score
from river.datasets import synth


n_features = 4
n_classes = 4
decay_horizon = 10000
total_instances = 10000


dataset = synth.RandomRBFDrift(
    seed_model=42,
    seed_sample=42,
    n_classes=n_classes,
    n_features=n_features,
    n_centroids=20,
    change_speed=0.5,
    n_drift_centroids=10,
)

clustream = cluster.CluStream(
    n_macro_clusters=24,
    max_micro_clusters=50,
    time_gap=1000,
    seed=42,
    halflife=0.4
)
decay_dataset_X = []
decay_dataset_y = []
i = 1

dataset = Insects().take(90000)

for x, y in dataset:
    clustream = clustream.learn_one(x)
    decay_dataset_X.append(x)
    decay_dataset_y.append(y)
    if i % decay_horizon == 0:
        predict_lables = [clustream.predict_one(x) for x in decay_dataset_X]
        purity = purity_score(decay_dataset_y, predict_lables)
        f1_score_r = F1_score_R(decay_dataset_y, predict_lables)
        f1_score_p = F1_score_P(decay_dataset_y, predict_lables)
        print("Purity:{}, F1_score_r:{}, F1_score_p:{}".format(purity, f1_score_r, f1_score_p))

        decay_dataset_X = []
        decay_dataset_y = []
    i += 1
