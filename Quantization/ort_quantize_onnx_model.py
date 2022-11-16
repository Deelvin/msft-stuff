import argparse

from onnxruntime.quantization import (
  quantize_static,
  quantize_dynamic,
  QuantType,
  QuantFormat,
)


if __name__ == "__main__":
  class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

  parser = argparse.ArgumentParser(
    description="Tool for quantization onnx model and saving result. ONNX Runtime quantization tool is used. " +
                "See details https://onnxruntime.ai/docs/performance/quantization.html and " +
                "samples https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization",
    formatter_class=MyFormatter
  )

  parser.add_argument("-m", "--model_path", default="", type=str, help=\
    "The path to the model in onnx format")
  parser.add_argument("-o", "--output_model_path", default="./quantized_model.onnx", type=str, help=\
    "The path to the model in onnx format")
  parser.add_argument("-qt", "--quantization_type", default="static", type=str, help=\
    "Quantization type: static or dynamic")
  parser.add_argument("-opt", "--optimize", action="store_true", default=False, help=\
    "Switch on optimization during quantization.")

  args = parser.parse_args()

  # Quantize and Save
  if args.quantization_type == "static":
    quantize_static(
      args.model_path,
      args.output_model_path,
      quant_format=QuantFormat.QDQ,
      activation_type=QuantType.QInt8,
      weight_type=QuantType.QInt8,
      optimize_model=args.optimize,
    )
  elif args.quantization_type == "dynamic":
    quantize_dynamic(
      args.model_path,
      args.output_model_path,
      weight_type=QuantType.QInt8,
      optimize_model=args.optimize,
    )
  else:
    print("Wrong quantization type:", args.quantization_type)
