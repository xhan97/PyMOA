# Copyright 2023 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import os
import sys
import unittest
import numpy as np

# noinspection PyProtectedMember
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.preprocessing import MinMaxScaler

# temporary solution for relative imports in case pyod is not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print(sys.path)

from models.denstream import DenStreamWithDBSCAN
from metrics.metrics import F1_score_P, F1_score_R, purity_score


class TestDenStream(unittest.TestCase):
    def setUp(self):
        self.data, self.lables = load_wine(return_X_y=True)

        self.clf = DenStreamWithDBSCAN(
            dimensions=len(self.data[0]),
            num_classes=3,
            window_range=30,
            epsilon=0.2,
            beta=0.2,
            mu=0.2,
            number_intialization_points=50,
            offline_multiplier=2.0,
            lambda_=0.05,
            processing_speed=1,
        )

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.data)
        print(pred_labels)
        print(self.lables)
        purity = purity_score(self.lables, pred_labels)
        f1_score_r = F1_score_R(self.lables, pred_labels)
        f1_score_p = F1_score_P(self.lables, pred_labels)
        print(purity)
        print(f1_score_p)
        print(f1_score_r)


if __name__ == "__main__":
    unittest.main()
