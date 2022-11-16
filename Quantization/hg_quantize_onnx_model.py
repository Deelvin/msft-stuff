import argparse

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig


def get_qconfig(target, is_static = False):
  if target == "arm64":
    return AutoQuantizationConfig.arm64(is_static=is_static, per_channel=False)
  elif target == "avx2":
    return AutoQuantizationConfig.avx2(is_static=is_static, per_channel=False)
  elif target == "avx512":
    return AutoQuantizationConfig.avx512(is_static=is_static, per_channel=False)
  elif target == "avx512_vnni":
    return AutoQuantizationConfig.avx512_vnni(is_static=is_static, per_channel=False)
  elif target == "tensorrt":
    return AutoQuantizationConfig.tensorrt(is_static=is_static, per_channel=False)
  else:
    print("Wrong target for quantization config:", target)
    return None

def main():
  class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

  parser = argparse.ArgumentParser(
    description="Tool for quantization onnx model and saving result. Hugging face quantization tool is used. " +
                "It bases on ONNX Runtime. See details https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization.",
    formatter_class=MyFormatter
  )

  parser.add_argument("-m", "--model_path", default="", type=str, help=\
    "The path to the model in onnx format")
  parser.add_argument("-o", "--output_model_dir", default="./", type=str, help=\
    "The parent directory for output quantized model in onnx format")
  parser.add_argument("-qt", "--quantization_type", default="static", type=str, help=\
    "Quantization type: static or dynamic")
  parser.add_argument("-t", "--target", default="avx2", type=str, help=\
    "HW target, it is needed for autoconfiguration args for quantization")
  parser.add_argument("-opt", "--optimize", action="store_true", default=False, help=\
    "Switch on optimization during quantization.")

  args = parser.parse_args()

  # Get original model
  quantizer = ORTQuantizer.from_pretrained(args.model_path)

  # Quantize and Save
  if args.quantization_type == "static":
    # Configure
    dqconfig = get_qconfig(args.target, True)
    # Quantize
    quantizer.quantize(
      save_dir=args.output_model_dir,
      file_suffix = "quantized",
      quantization_config=dqconfig,
      calibration_tensors_range={}, # ?
    )
  elif args.quantization_type == "dynamic":
    # Configure
    dqconfig = get_qconfig(args.target)
    # Quantize
    quantizer.quantize(
      save_dir=args.output_model_dir,
      file_suffix = "dyn_quantized",
      quantization_config=dqconfig,
    )
  else:
    print("Wrong quantization type:", args.quantization_type)

if __name__ == "__main__":
  main()
