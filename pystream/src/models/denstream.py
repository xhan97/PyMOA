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

_LOGGER = setup_logger("dstream-with-dbscan", logging.DEBUG)


from base import BaseClustering
from base import _IMPORTS

_IMPORTS = _IMPORTS + [
    "moa.clusterers.denstream.WithDBSCAN",
]


class DenStreamWithDBSCAN(BaseClustering):
    """Clustream using Kmeans."""

    def __init__(
        self,
        dimensions: int,
        num_classes: int = 3,
        window_range: int = 1000,
        epsilon: float = 0.02,
        beta: float = 0.2,
        mu: float = 1.0,
        number_intialization_points: int = 1000,
        offline_multiplier: float = 2.0,
        lambda_: float = 0.25,
        processing_speed: int = 100,
    ) -> None:
        super().__init__(dimensions, num_classes)
        self._window_range = window_range
        self._epsilon = epsilon
        self._beta = beta
        self._mu = mu
        self._number_intialization_points = number_intialization_points
        self._offline_multiplier = offline_multiplier

        self._lambda_ = lambda_
        self._processing_speed = processing_speed
        self._gateway = setup_java_gateway(imports=_IMPORTS)
        self._header = self._generate_header()
        self._clusterer = self._initialize_clusterer()

    @property
    def window_range(self) -> float:
        """Get horizon window range."""
        return self._window_range

    @property
    def epsilon(self) -> float:
        """Get Epsilon neighbourhood."""
        return self._epsilon

    @property
    def beta(self) -> float:
        """Get DBSCAN beta parameter."""
        return self._beta

    @property
    def mu(self) -> float:
        """Get DBSCAN mu parameter."""
        return self._mu

    @property
    def number_intialization_points(self) -> float:
        """Get Number of points to used for initialization."""
        return self._number_intialization_points

    @property
    def offline_multiplier(self) -> float:
        """Get offline multiplier for epsilion."""
        return self._offline_multiplier

    @property
    def lambda_(self) -> float:
        """Get DBSCAN lambda parameter."""
        return self._lambda_

    @property
    def processing_speed(self) -> float:
        """Get processing speed per time unit."""
        return self._processing_speed

    def _initialize_clusterer(self) -> Any:
        clusterer = self._gateway.jvm.WithDBSCAN()

        clusterer.horizonOption.setValue(self._window_range)
        clusterer.epsilonOption.setValue(self._epsilon)
        clusterer.betaOption.setValue(self._beta)
        clusterer.muOption.setValue(self._mu)
        clusterer.initPointsOption.setValue(self._number_intialization_points)
        clusterer.offlineOption.setValue(self._offline_multiplier)
        clusterer.lambdaOption.setValue(self._lambda_)
        clusterer.speedOption.setValue(self._processing_speed)

        clusterer.resetLearning()

        return clusterer
