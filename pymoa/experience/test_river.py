from river import cluster
from river.datasets import Insects
# import matplotlib.pyplot as plt

import os
import sys

# temporary solution for relative imports in case pyod is not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print(sys.path)


from metrics.metrics import F1_score_P, F1_score_R, purity_score
from river.datasets import synth


n_features = 2
n_classes = 5
decay_horizon = 1000
total_instances = 10000


# dataset = synth.RandomRBFDrift(
#     seed_model=42,
#     seed_sample=42,
#     n_classes=n_classes,
#     n_features=n_features,
#     n_centroids=5,
#     change_speed=0.0,
#     n_drift_centroids=1,
# )

# clustream = cluster.STREAMKMeans(
#     chunk_size=100, n_clusters=5, halflife=0.5, sigma=0.5, seed=42)

clustream = cluster.CluStream(
    n_macro_clusters=5,
    max_micro_clusters=20,
    time_gap=100,
    seed=42,
    halflife=0.05,
    micro_cluster_r_factor=5,
)
decay_dataset_X = []
decay_dataset_y = []
i = 1

dataset = Insects().take(10000)


# def plot_sample(x, y):
#     plt.scatter(x, y)
#     plt.show()


for x, y in dataset:
    clustream = clustream.learn_one(x)
    decay_dataset_X.append(x)
    decay_dataset_y.append(y)
    if i % decay_horizon == 0:
        predict_lables = [clustream.predict_one(x) for x in decay_dataset_X]
        purity = purity_score(decay_dataset_y, predict_lables)
        f1_score_r = F1_score_R(decay_dataset_y, predict_lables)
        f1_score_p = F1_score_P(decay_dataset_y, predict_lables)
        print(
            "Purity:{}, F1_score_r:{}, F1_score_p:{}".format(
                purity, f1_score_r, f1_score_p
            )
        )
        decay_dataset_X = []
        decay_dataset_y = []
    i += 1
