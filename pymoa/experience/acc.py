# Copyright (C) 2023 Xin Han
#
# This file is part of PyMOA.
#
# PyMOA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyMOA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyMOA.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys

# temporary solution for relative imports in case pyod is not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print(sys.path)

from data.dataloader.river_loader import RiverLoader
from metrics.metrics import F1_score_P, F1_score_R, purity_score
from models.clustering.clustream import Clustream
from models.clustering.streamkm import StreamKM
from river.datasets import synth

n_features = 4
n_classes = 4
decay_horizon = 1000
total_instances = 10000


dataset = synth.RandomRBFDrift(
    seed_model=42,
    seed_sample=42,
    n_classes=n_classes,
    n_features=n_features,
    n_centroids=20,
    change_speed=0.87,
    n_drift_centroids=10,
)

dataset = RiverLoader(dataset, total_instances=total_instances, decay_horizon=decay_horizon)

clf = Clustream(
    dimensions=n_features,
    num_classes=n_classes,
    time_window=200,
    max_num_kernels=10,
    kernel_radius=3,
    k=3,
)

# clf = StreamKM(
#     dimensions=n_features,
#     num_classes=n_classes,
#     size_coreset=200,
#     number_clusters=3,
#     #length=1000,
#     random_seed=42,
# )

for i, (x, y) in enumerate(dataset, start=1):
    #clf.learn_one(x)
    if i % decay_horizon == 0:
        decay_dataset = dataset.decay_dataset
        #predict_lables = clf.predict_batch(decay_dataset['X'])
        predict_lables = clf.fit_predict(decay_dataset['X'])
        clf = clf._reset_instances()
        #predict_lables = clf.predict_batch(decay_dataset['X'])
        purity = purity_score(decay_dataset['y'], predict_lables)
        f1_score_r = F1_score_R(decay_dataset['y'], predict_lables)
        f1_score_p = F1_score_P(decay_dataset['y'], predict_lables)
        print("Purity:{}, F1_score_r:{}, F1_score_p:{}".format(purity, f1_score_r, f1_score_p))

# print(len(dataset))
# print(dataset.decay_dataset)
