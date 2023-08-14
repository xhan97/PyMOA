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
    "moa.clusterers.clustree.ClusTree",
]


class Clustree(BaseClustering):
    def __init__(
        self,
        dimensions: int,
        num_classes: int = 3,
        window_range: int = 1000,
        max_height: int = 8,
    ) -> None:
        super().__init__(dimensions, num_classes)
        self._window_range = window_range
        self._max_height = max_height
        self._gateway = setup_java_gateway(imports=_IMPORTS)
        self._header = self._generate_header()
        self._clusterer = self._initialize_clusterer()

    @property
    def window_range(self) -> float:
        """Get window range."""
        return self._window_range

    @property
    def max_height(self) -> float:
        """
        Get max tree height
        """
        return self._max_height

    def _initialize_clusterer(self) -> Any:
        clusterer = self._gateway.jvm.ClusTree()

        clusterer.horizonOption.setValue(self._window_range)
        clusterer.maxHeightOption.setValue(self._max_height)

        clusterer.resetLearning()

        return clusterer
