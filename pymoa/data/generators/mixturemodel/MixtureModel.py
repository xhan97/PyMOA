import numpy as np
from scipy.stats import multivariate_normal
import math


class MixtureModel:
    def __init__(
        self, num_classes, num_attributes, instance_random_seed, model_random_seed
    ):
        self.num_models = num_classes
        self.dimensions = num_attributes
        self.weights = np.zeros(num_classes)
        self.model_array = []
        self.model_random = np.random.RandomState(model_random_seed)
        self.instance_random = np.random.RandomState(instance_random_seed)
        self.range = num_classes

        self._initialize_models()

    def _initialize_models(self):
        weight_sum = 0
        means = np.zeros(self.dimensions)

        for i in range(self.num_models):
            self.weights[i] = self.model_random.rand()
            weight_sum += self.weights[i]

            for j in range(self.dimensions):
                means[j] = (self.model_random.rand() * self.range) - (self.range / 2.0)

            covariances = self._generate_covariance(self.dimensions)
            self.model_array.append(multivariate_normal(means, covariances))

        self._normalize_weights(weight_sum)

    def _generate_covariance(self, d):
        x = np.random.rand(d, d)
        covariances = np.zeros((d, d))
        matrix_sum = 0

        for j in range(d):
            for k in range(d):
                x[j][k] = (
                    ((self.model_random.rand() * 2.0) - 1.0)
                    + ((self.model_random.rand() * 2.0) - 1.0)
                ) / 2.0

        for j in range(d):
            for k in range(j + 1):
                for l in range(d):
                    matrix_sum += x[j][l] * x[k][l]

                covariances[j][k] = matrix_sum
                covariances[k][j] = matrix_sum
                matrix_sum = 0

        return covariances

    def _normalize_weights(self, weight_sum):
        for i in range(self.num_models):
            self.weights[i] /= weight_sum

    def next_instance(self):
        index = self.instance_random.choice(np.arange(self.num_models), p=self.weights)
        point = self.model_array[index].rvs()
        y = index
        return point, y

    def density_at(self, point):
        density = 0

        for i in range(self.num_models):
            density += self.weights[i] * self.model_array[i].pdf(point)

        return density

    def restart(self, instance_random_seed, model_random_seed):
        self.instance_random.seed(instance_random_seed)
        self.model_random.seed(model_random_seed)

    def get_dimensions(self):
        return self.dimensions

    def get_num_models(self):
        return self.num_models

    def get_weight(self, i):
        return self.weights[i]

    def get_weights(self):
        return self.weights

    def get_means(self, i):
        return self.model_array[i].mean

    def get_covariance(self, i):
        return self.model_array[i].cov

    def set_means(self, i, new_means):
        mvndist = multivariate_normal(new_means, self.model_array[i].cov)
        self.model_array[i] = mvndist

    def to_string(self):
        sb = ""


class MixtureModelWithModel(MixtureModel):
    def __init__(
        self,
        num_classes,
        num_attributes,
        instance_random_seed,
        model_random_seed,
        mm1: MixtureModel,
        targetDist: float,
    ):
        super().__init__(
            num_classes, num_attributes, instance_random_seed, model_random_seed
        )
        self.mm1 = mm1
        self.targetDist = targetDist
        self._adjust_weight()

    def _adjust_weight(self):
        weight_sum = 0.0
        adjustment_factor = math.pow(self.targetDist, 2.0)

        for i in range(self.num_models):
            # Adjust weights and means
            if i < self.mm1.get_num_models():
                self.weights[i] = (self.weights[i] * adjustment_factor) + (
                    self.mm1.get_weight(i) * (1.0 - adjustment_factor)
                )
                new_means = [0.0] * self.dimensions

                for j in range(self.dimensions):
                    new_means[j] = (self.get_means(i)[j] * adjustment_factor) + (
                        self.mm1.get_means(i)[j] * (1.0 - adjustment_factor)
                    )
                self.set_means(i, new_means)

            weight_sum += self.weights[i]

        self._normalize_weights(weight_sum)
