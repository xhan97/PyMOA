import math
import random
from numba import jit

from MixtureModel import MixtureModel, MixtureModelWithModel


class ConceptMixtureModel:
    def __init__(
        self,
        num_atts,
        num_classes_pre,
        burn_in_instances,
        drift_duration,
        drift_magnitude,
        precision_drift_magnitude,
        num_classes_post,
        model_random_seed,
        instance_random_seed,
    ):
        """
        :param num_atts: The number of attributes to generate.
        :param num_classes_pre: The number of classes in the data stream and the number of models to include in the mixture model pre-concept drift.
        :param burn_in_instances: The number of instances to draw from the pre-concept drift mixture model.
        :param drift_duration: The number of instances between the stable pre-drift mixture model and the stable post-drift mixture model.
        :param drift_magnitude: Magnitude of the drift between the starting probability and the one after the drift. [0,1].
        :param precision_drift_magnitude: Precision of the drift magnitude for p(x) (how far from the set magnitude is acceptable)
        :param num_classes_post: The number of classes in the data stream and the number of models to include in the mixture model post-concept drift.
        :param model_random_seed:  Seed for random generation of model.
        :param instance_random_seed: Seed for random generation of instances.
        """
        self.num_atts = num_atts
        self.num_classes_pre = num_classes_pre
        self.burn_in_instances = burn_in_instances
        self.drift_duration = drift_duration
        self.drift_magnitude = drift_magnitude
        self.instance_random_seed = instance_random_seed
        self.precision_drift_magnitude = precision_drift_magnitude
        self.num_classes_post = num_classes_post
        self.model_random_seed = model_random_seed
        self._initialize()

    @jit
    def _initialize(self):
        print("Initialized...")
        self._num_instances = 0
        self._last_instance_pre = self.burn_in_instances
        self._first_instance_post = self._last_instance_pre + self.drift_duration + 1
        self._monte_carlo_random = random.Random()
        self._integrate_range = max(self.num_classes_pre, self.num_classes_pre) + 4.0
        y = 0

        while True:
            self._mixture_model_pre = MixtureModel(
                self.num_classes_pre,
                self.num_atts,
                self.instance_random_seed + y,
                self.model_random_seed + y,
            )
            y += 1
            z = y

            while True:
                self._mixture_model_post = MixtureModelWithModel(
                    self.num_classes_post,
                    self.num_atts,
                    self.instance_random_seed + z,
                    self.model_random_seed + z,
                    self._mixture_model_pre,
                    self.drift_magnitude,
                )
                h_dist = self._hellinger_distance(
                    self._mixture_model_pre,
                    self._mixture_model_post,
                    self.drift_magnitude,
                )
                dist_miss = h_dist - self.drift_magnitude
                print(
                    "{}.{}: The Hellinger distance was evaluated as {},compared against the desired range {} +/- {}".format(
                        y - 1,
                        z - 1,
                        h_dist,
                        self.drift_magnitude,
                        self.precision_drift_magnitude,
                    )
                )

                if (z <= 100) and (abs(dist_miss) > self.precision_drift_magnitude):
                    z += 1
                else:
                    break

            if abs(dist_miss) > self.precision_drift_magnitude:
                break

        print(
            "The Hellinger distance was evaluated as {}, compared against the desired range {} +/- {}".format(
                h_dist, self.drift_magnitude, self.precision_drift_magnitude
            )
        )

        print("The distance was off by {}".format(dist_miss))

    def next_instance(self):
        self._num_instances += 1

        # Post concept drift model
        if self._num_instances > self._first_instance_post:
            return self._mixture_model_post.next_instance()

        # If first instance post, generate header and return
        # next instance from mixture model post
        if self._num_instances == self._first_instance_post:
            # self.generate_header(self.num_classes_post)
            return self._mixture_model_post.next_instance()

        # Pre concept drift model
        elif self._num_instances <= self.last_instance_pre:
            return self._mixture_model_pre.next_instance()

        # During concept drift mix of models
        else:
            if self._monte_carlo_random.next_double() < (
                (self._num_instances - self.burn_in_instances) / (self.drift_duration)
            ):
                return self._mixture_model_post.next_instance()
            # Otherwise, return next instance from mixture model pre
            else:
                return self._mixture_model_pre.next_instance()

    @jit
    def _hellinger_distance(
        self, mm1: MixtureModel, mm2: MixtureModel, target_dist: float
    ):
        monte_carlo = 0.0
        running_sum = 0.0
        hellinger_distance = -1.0

        volume = math.pow(self._integrate_range, self.num_atts)
        error = float("inf")
        N = 0.0
        sample_var = 0.0
        mean = 0.0
        M2 = 0.0
        delta1 = 0.0
        delta2 = 0.0
        x = 0.0
        point = [0.0] * self.num_atts
        self._monte_carlo_random.seed(
            self.instance_random_seed + self.model_random_seed
        )

        while error > 0.001:
            # Randomly generate the point at which to evaluate the function
            for i in range(self.num_atts):
                point[i] = (
                    self._monte_carlo_random.random() * self._integrate_range
                ) - (self._integrate_range / 2.0)

            # Evaluate the function at point and add the result to the running sum
            x = math.sqrt(mm1.density_at(point) * mm2.density_at(point))
            running_sum += x
            N += 1

            # Adjust other values of interest
            delta1 = x - mean
            mean += delta1 / N
            delta2 = x - mean
            M2 += delta1 * delta2
            monte_carlo = volume * running_sum / N

            # Once a sufficient base of samples has been built,
            # calculate the sample variance and estimate the error
            if N > 1000000:
                sample_var = M2 / (N - 1)
                error = volume * math.sqrt(sample_var) / math.sqrt(N)
                hellinger_distance = math.sqrt(1.0 - monte_carlo)

                # If the target distance is no longer within the error margin
                # around the estimated distance then break from the WHILE loop
                if abs(target_dist - hellinger_distance) > (math.sqrt(error)):
                    break

        # print("N: ", N, ", monte_carlo: ", monte_carlo, ", 1.0 - monte_carlo: ", (1.0-monte_carlo), ", and error: ", error)
        # print("Hellinger distance is estimated as (", hellinger_distance, " +/- ", math.sqrt(error), "); (target distance was ", target_dist, ")")
        return hellinger_distance


if __name__ == "__main__":
    cmm = ConceptMixtureModel(
        num_atts=2,
        num_classes_pre=3,
        burn_in_instances=10,
        drift_duration=1000,
        precision_drift_magnitude=0.01,
        num_classes_post=4,
        model_random_seed=42,
        instance_random_seed=42,
        drift_magnitude=0.5,
    )
    print(cmm.next_instance())
