import time
import typing

import numpy


def milliseconds_to_seconds(time_ms: float) -> float:
    return float(time_ms / 1000)


def seconds_to_milliseconds(time_seconds: float) -> float:
    return float(time_seconds * 1000)


def measure_function(
    num_trials: int,
    num_runs_per_trial: int,
    warmup_ms: int,
    func: typing.Callable,
) -> typing.List[float]:
    start = time.perf_counter()
    print("Running model for warmup")
    func()
    while time.perf_counter() - start < milliseconds_to_seconds(warmup_ms):
        func()

    times = []
    print("Running benchmark")
    for _ in range(num_trials):
        start = time.perf_counter()
        for _ in range(num_runs_per_trial):
            func()
        finish = time.perf_counter()

        times.append((finish - start) / num_runs_per_trial)

    return times


class ModelBase:
    def benchmark(
        self, warmup_ms: int = 500, num_trials: int = 30, num_runs_per_trial: int = 1
    ) -> typing.Tuple[float, float]:
        inputs = self.prepare_input()
        run_fn = self.get_inference_function(inputs)

        runtimes_s = measure_function(num_trials, num_runs_per_trial, warmup_ms, run_fn)
        runtimes_ms = list(map(seconds_to_milliseconds, runtimes_s))
        mean = numpy.mean(runtimes_ms)
        std = numpy.std(runtimes_ms)

        print(f"Measured model in ms: mean {mean}, std {std}")
        return float(mean), float(std)

    def get_inference_function(self, inputs: typing.Dict[str, numpy.ndarray]):
        ...

    def prepare_input(self) -> typing.Any:
        ...
