import time

sklearn_classifiers = [
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
]


sklearn_regressors = [
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
]


class Profiler:
    def __init__(self, name="", show_latency=False, iterations_number=1):
        self.name = name
        self.show_latency = show_latency
        self.iterations_number = iterations_number

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, type, value, traceback):
        end_time = time.perf_counter()
        if self.show_latency:
            print(
                "[{}] elapsed time: {:.3f} ms ({} iterations)".format(
                    self.name,
                    (end_time - self.start_time) * 1000 / self.iterations_number,
                    self.iterations_number,
                )
            )
