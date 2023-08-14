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

import abc
from typing import Any, Dict, Iterable, List
import numpy as np

import six
from py4j.java_gateway import JavaGateway
from sklearn.base import BaseEstimator, ClusterMixin

_IMPORTS = [
    "com.yahoo.labs.samoa.instances.SparseInstance",
    "com.yahoo.labs.samoa.instances.DenseInstance",
    "com.yahoo.labs.samoa.instances.Instance",
    "com.yahoo.labs.samoa.instances.Instances",
    "com.yahoo.labs.samoa.instances.InstanceStream",
    "com.yahoo.labs.samoa.instances.InstancesHeader",
    "com.yahoo.labs.samoa.instances.Attribute",
    "moa.core.FastVector",
    "moa.clusterers.KMeans",
    "moa.gui.visualization.DataPoint",
    "moa.cluster.Clustering",
]


@six.add_metaclass(abc.ABCMeta)
class BaseClustering(BaseEstimator, ClusterMixin):
    """Base clustering using MOA."""

    def __init__(
        self,
        dimensions: int,
        num_classes: int = 3,
    ) -> None:
        """
        Initialize clusterer.

        :param dimensions: Data dimensionality.
        :param window_range: Horizon window range.
        """
        self._dimensions = dimensions

        self._num_classes = num_classes

        self._gateway: JavaGateway = None

        self._header: Any = None
        self._clusterer: Any = None
        self._instances: List[Any] = []
        self._found_clustering: Any = None
        self._m_timestamp = 0

    @property
    def dimension(self) -> float:
        """Get data dimentions."""
        return self._dimensions

    @property
    def num_classes(self) -> float:
        """Get classes number."""
        return self._num_classes

    # @property
    # def data_points(self) -> List[Any]:
    #     return self._data_points

    @property
    def gateway(self) -> JavaGateway:
        return self._gateway

    @property
    def found_clustering(self) -> Any:
        return self._found_clustering

    def _generate_header(self):
        """
        Generate header.

        Follows the same steps as:
        https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/streams/generators/RandomRBFGenerator.java#L154
        """
        gateway = self._gateway

        FastVector = gateway.jvm.FastVector
        Attribute = gateway.jvm.Attribute
        InstancesHeader = gateway.jvm.InstancesHeader
        Instances = gateway.jvm.Instances

        attributes = FastVector()
        for i in range(self._dimensions):
            attributes.addElement(Attribute(f"att {(i + 1)}"))

        classLabels = FastVector()
        for i in range(self._num_classes):
            classLabels.addElement("class" + str(i + 1))

        attributes.addElement(Attribute("class", classLabels))

        header = InstancesHeader(Instances("", attributes, 0))
        header.setClassIndex(header.numAttributes() - 1)

        return header

    def _initialize_clusterer(self) -> Any:
        """Initialize clusterer."""
        pass

    def _create_instance(self, vector: Iterable[float], y: int = None) -> Any:
        """Create instance."""
        instance = self._gateway.jvm.DenseInstance(float(self._dimensions))

        for index, number in enumerate(vector):
            instance.setValue(index, number)

        instance.setDataset(self._header)

        if y is not None:
            instance.setClassValue(float(y))

        return instance

    def _create_instances(self, X, lables):
        if lables is None:
            instances = [self._create_instance(vector) for vector in X]
        else:
            instances = [
                self._create_instance(vector, y=y) for vector, y in zip(X, lables)
            ]

        return instances

    def _reset_clusterering(self):
        self._found_clustering = None
        return self

    def _get_clustering(self):
        found_clustering = self._clusterer.getClusteringResult()
        # ground_true_clustering = self._gateway.jvm.Clustering(self._data_points)
        if self._clusterer.implementsMicroClusterer():
            micro_clustering = self._clusterer.getMicroClusteringResult()

            if self._clusterer.evaluateMicroClusteringOption.isSet():
                found_clustering = micro_clustering
            else:
                if found_clustering is None and micro_clustering is not None:
                    found_clustering = micro_clustering
                    # kmeans = self._gateway.jvm.KMeans()
                    # found_clustering = kmeans.gaussianMeans(
                    #     ground_true_clustering, micro_clustering
                    # )
        self._found_clustering = found_clustering
        return found_clustering

    def transform(self) -> None:
        """Transform."""
        raise NotImplementedError

    def learn_one(self, x: Iterable[float], y: int = None):
        """
        Partial fit model.

        :param X: An iterable of vectors.
        """
        instance = self._create_instance(x, y=y)
        point = self._gateway.jvm.DataPoint(instance, self._m_timestamp)
        if y is not None:
            instance.deleteAttributeAt(point.classIndex())
        self._clusterer.trainOnInstanceImpl(instance)
        self._m_timestamp += 1

        if self._found_clustering is not None:
            self._reset_clusterering()

        return self

    def learn_batch(self, X: Iterable[Iterable[float]], lables: Iterable[int] = None):
        """
        Partial fit model.

        :param X: An iterable of vectors.
        """

        if lables is None:
            for vector in X:
                self.learn_one(vector)
        else:
            for vector, y in zip(X, lables):
                self.learn_one(vector, y=y)

        return self

    def predict_one(self, x: Iterable[float], y: int = None) -> int:
        if self._found_clustering is None:
            found_clustering = self._get_clustering()
        else:
            found_clustering = self._found_clustering

        instance = self._create_instance(x)
        covered = False
        label = 0
        min_distance = np.inf
        for c in range(found_clustering.size()):
            prob = found_clustering.get(c).getInclusionProbability(instance)
            if prob >= 1:
                x_distance = found_clustering.get(c).getCenterDistance(instance)
                if x_distance <= min_distance:
                    label = c
                    min_distance = x_distance
                covered = True
        if not covered:
            label = -1
        return label

    def predict_batch(
        self, X: Iterable[Iterable[float]], lables: Iterable[int] = None
    ) -> List[int]:
        if lables is None:
            labels = [self.predict_one(vector) for vector in X]
        else:
            labels = [self.predict_one(vector, y=y) for vector, y in zip(X, lables)]
        return labels

    def fit_predict(
        self, X: Iterable[Iterable[float]], lables: Iterable[int] = None
    ) -> List[int]:
        """
        Partial fit model.

        :param X: An iterable of vectors.
        """
        clf = self.learn_batch(X, lables=lables)

        return clf.predict_batch(X, lables=lables)
