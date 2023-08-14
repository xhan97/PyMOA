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


from utils import setup_logger, setup_java_gateway

_LOGGER = setup_logger("clustree", logging.DEBUG)


from base import BaseClustering
from base import _IMPORTS

_IMPORTS = _IMPORTS + [
    "moa.clusterers.clustream.WithKmeans",
]


class Clustream(BaseClustering):
    """Clustream using Kmeans."""

    def __init__(
        self,
        dimensions: int,
        num_classes: int = 3,
        time_window: int = 1000,
        max_num_kernels: int = 100,
        kernel_radius: int = 2,
        k: int = 2,
    ) -> None:
        super().__init__(dimensions, num_classes)
        self._time_window = time_window
        self._max_num_kernels = max_num_kernels
        self._kernel_radius = kernel_radius
        self._k = k
        self._gateway = setup_java_gateway(imports=_IMPORTS)
        self._header = self._generate_header()
        self._clusterer = self._initialize_clusterer()

    @property
    def time_window(self) -> int:
        """Get horizon window range."""
        return self._time_window

    @property
    def max_num_kernels(self) -> float:
        """Get max num kernel."""
        return self._max_num_kernels

    @property
    def kernel_radius(self) -> float:
        """
        Get kernal radius.
        """
        return self._kernel_radius

    @property
    def k(self) -> float:
        """
        Get k.
        """
        return self._k

    def _initialize_clusterer(self) -> Any:
        clusterer = self._gateway.jvm.WithKmeans()

        clusterer.timeWindowOption.setValue(self.time_window)
        clusterer.kOption.setValue(self.k)
        clusterer.maxNumKernelsOption.setValue(self.max_num_kernels)
        clusterer.kernelRadiFactorOption.setValue(self.kernel_radius)

        clusterer.resetLearning()

        return clusterer
