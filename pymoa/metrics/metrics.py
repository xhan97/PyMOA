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
from typing import Any, Dict, Iterable, List

import numpy as np
import six
from py4j.java_gateway import JavaGateway
from py4j.java_gateway import java_import
from sklearn import metrics
import numpy as np


def get_contingency_matrix(y_true, y_pred):
    """
    Compute contingency matrix (also called confusion matrix),
    shape=[n_classes_true, n_classes_pred]
    """
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return contingency_matrix


def purity_score(y_true, y_pred):
    ctm = get_contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(ctm, axis=0)) / np.sum(ctm)


def F1_score_P(y_true, y_pred):
    """
    F1 as defined in P3C, try using F1 optimization
    """
    ctm = get_contingency_matrix(y_true, y_pred)

    max_index = np.argmax(ctm, axis=0)
    F1_P = 0.0

    for j in range(ctm.shape[1]):
        precision = ctm[max_index[j], j] / sum(ctm[:, j])
        recall = ctm[max_index[j], j] / sum(ctm[max_index[j], :])
        f1 = 0.0
        if precision > 0 and recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        F1_P += f1
    F1_P /= ctm.shape[1]
    return F1_P


def F1_score_R(y_true, y_pred):
    """
    F1 as defined in .... mainly maximizes F1 for each class"""
    ctm = get_contingency_matrix(y_true, y_pred)

    F1_R = 0.0
    for i in range(ctm.shape[0]):
        max_f1 = 0.0
        for j in range(ctm.shape[1]):
            precision = ctm[i, j] / sum(ctm[:, j])
            recall = ctm[i, j] / sum(ctm[i, :])
            f1 = 0.0
            if precision > 0 and recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            if f1 > max_f1:
                max_f1 = f1
        F1_R += max_f1
    F1_R /= ctm.shape[0]

    return F1_R


def cmm_score(gateway: JavaGateway, clustering, gt_clustering, points: Iterable[float]):
    cmm_path = "moa.evaluation.CMM"
    java_import(gateway.jvm, cmm_path)
    cmm = gateway.jvm.CMM()
    cmm.evaluateClustering(clustering, gt_clustering, points)
    cmm_score = {}
    for i in range(cmm.getNumMeasures()):
        cmm_score[cmm.getName(i)] = cmm.getAllValues(i)
    return cmm_score


if __name__ == "__main__":
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([1, 1, 0, -1, 2, 2])
    print(purity_score(y_true, y_pred))
    print(F1_score_P(y_true, y_pred))
    print(F1_score_R(y_true, y_pred))
