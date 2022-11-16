from functools import partial

import onnxruntime

def ort_test(benchmark_test, model_path, ort_inputs):
  print("----- ONNXruntime testing -----")
  provider_name = ["CPUExecutionProvider"]
  # Run the model on the backend
  ort_session = onnxruntime.InferenceSession(model_path, providers=provider_name)

  ort_runner = partial(ort_session.run, [], ort_inputs)

  benchmark_test(ort_runner, framework_name = "ONNX Runtime")
