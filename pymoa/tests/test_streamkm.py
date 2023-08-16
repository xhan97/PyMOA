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

from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.preprocessing import MinMaxScaler

# temporary solution for relative imports in case pyod is not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print(sys.path)

from models.clustering.streamkm import StreamKM
from metrics.metrics import F1_score_P, F1_score_R, purity_score
from IsoKernel import IsoKernel
import time


class TestDenStream(unittest.TestCase):
    def setUp(self):
        self.data, self.lables = load_wine(return_X_y=True)
        self.data = MinMaxScaler().fit_transform(self.data)
        
        # ik = IsoKernel(n_estimators=200, max_samples=8)
        # st_time = time.time()
        # self.data = ik.fit_transform(self.data)
        # et_time = time.time()
        # print("IsoKernel time: ", et_time - st_time)
        self.clf = StreamKM(
            dimensions=len(self.data[0]),
            num_classes=3,
            size_coreset=60,
            number_clusters=3,
            length=1000,
            random_seed=42,
        )

    def test_fit_predict(self):
        st_time = time.time()
        pred_labels = self.clf.fit_predict(self.data)
        et_time = time.time()
        print("StreamKM time: ", et_time - st_time)
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
