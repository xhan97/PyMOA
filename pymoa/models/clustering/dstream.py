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
    "moa.clusterers.dstream.Dstream",
]


class DStream(BaseClustering):
    def __init__(
        self,
        dimensions: int,
        num_classes: int = 3,
        decay_factor: float = 0.998,
        cm: float = 3.0,
        cl: float = 0.8,
        beta: float = 0.3,
    ) -> None:
        super().__init__(dimensions, num_classes)
        self._decay_factor = decay_factor
        self._cm = cm
        self._beta = beta
        self._cl = cl

        self._gateway = setup_java_gateway(imports=_IMPORTS)
        self._header = self._generate_header()
        self._clusterer = self._initialize_clusterer()

    @property
    def decay_factor(self) -> float:
        """Get window range."""
        return self._decay_factor

    @property
    def cm(self) -> float:
        """
        Get max tree height
        """
        return self._cm

    @property
    def beta(self) -> float:
        """
        Get beata"""
        return self._beta

    @property
    def cl(self) -> float:
        """
        Get cl"""
        return self._cl

    def _initialize_clusterer(self) -> Any:
        clusterer = self._gateway.jvm.Dstream()

        clusterer.decayFactorOption.setValue(self.decay_factor)
        clusterer.cmOption.setValue(self.cm)
        clusterer.clOption.setValue(self.cl)
        clusterer.betaOption.setValue(self.beta)
        
        clusterer.resetLearning()

        return clusterer
