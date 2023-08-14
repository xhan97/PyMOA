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

import logging
from typing import Dict, List, Iterable, Any


from pymoa.utils.utils import setup_logger, setup_java_gateway

_LOGGER = setup_logger("clustree", logging.DEBUG)


from .base import BaseClustering
from .base import _IMPORTS

_IMPORTS = _IMPORTS + [
    "moa.clusterers.streamkm.StreamKM",
    "moa.cluster.Clustering",
]


class StreamKM(BaseClustering):
    def __init__(
        self,
        dimensions: int,
        num_classes: int = 3,
        size_coreset: int = 10000,
        number_clusters: int = 5,
        length: int = 100000,
        random_seed: int = 1,
    ) -> None:
        super().__init__(dimensions, num_classes)
        self._size_coreset = size_coreset
        self._number_clusters = number_clusters
        self._length = length
        self._random_seed = random_seed
        self._gateway = setup_java_gateway(imports=_IMPORTS)
        self._header = self._generate_header()
        self._clusterer = self._initialize_clusterer()

    @property
    def size_coreset(self) -> float:
        """Get coreset size."""
        return self._size_coreset

    @property
    def number_clusters(self) -> float:
        """Get clusters number."""
        return self._number_clusters

    @property
    def length(self) -> float:
        """Get length parameter."""
        return self._length

    @property
    def random_seed(self) -> float:
        """Get  random seed parameter."""
        return self._random_seed

    def _initialize_clusterer(self) -> Any:
        clusterer = self._gateway.jvm.StreamKM()

        clusterer.sizeCoresetOption.setValue(self._size_coreset)
        clusterer.numClustersOption.setValue(self._number_clusters)
        clusterer.lengthOption.setValue(self._length)
        clusterer.randomSeedOption.setValue(self._random_seed)

        clusterer.resetLearning()

        return clusterer


    def get_f1_score(self):
        return self.f1_score(
            self._found_clustering, self.gt_clustering, self.data_points
        )
        
    def get_cmm_score(self):
        return self.cmm_score(
            self._found_clustering, self.gt_clustering, self.data_points
        )
