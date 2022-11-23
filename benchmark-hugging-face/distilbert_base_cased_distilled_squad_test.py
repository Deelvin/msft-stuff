import argparse
from functools import partial

import onnx

from utils import perf_test, get_distilbert_inputs

if __name__ == "__main__":
  class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

  parser = argparse.ArgumentParser(
    description="Performance test of Distilbert-base-cased distilled by SQuAD model on TVM using VirtualMachine. " +
                "Hugging face model zoo: https://huggingface.co/distilbert-base-cased-distilled-squad?context=My+name+is+Wolfgang+and+I+live+in+Berlin&question=Where+do+I+live%3F, " +
                "ONNX format: https://huggingface.co/philschmid/distilbert-onnx?context=My+name+is+Wolfgang+and+I+live+in+Berlin&question=Where+do+I+live%3F. " +
                "Default target HW is Intel® Xeon® Scalable Platinum 8173M (skylake-avx512).",
    formatter_class=MyFormatter
  )
  # Model format
  parser.add_argument("-m", "--model_path", default="", type=str, help=\
    "The path to Distilbert-base-cased distilled by SQuAD model in onnx format")
  parser.add_argument("-q", "--input_question", default="Who was Jim Henson?", type=str, help=\
    "Test input question for Distilbert-base-cased distilled by SQuAD model")
  parser.add_argument("-i", "--input_text", default="Jim Henson was a nice puppet", type=str, help=\
    "Test input text for Distilbert-base-cased distilled by SQuAD model")
  parser.add_argument("-tvm", action="store_false", default=True, help=\
    "Performance test of TVM")
  parser.add_argument("-ort", action="store_true", default=False, help=\
    "Performance test of ONNX Runtime")
  parser.add_argument("-t", "--target", default="llvm -mcpu=skylake-avx512", type=str, help=\
    "Target for model inference")
  parser.add_argument("-n", "--iters_number", default=1000, type=int, help=\
    "Number of iterations of inference for performance measurement")
  parser.add_argument("-a", "--artificial_input", action="store_true", default=False, help=\
    "Artificially generated inputs. if false the default text from utils is tokenized")

  args = parser.parse_args()

  opt_level = 3
  freeze = True

  onnx_model = onnx.load(args.model_path)

  encoded_inputs = get_distilbert_inputs(args.artificial_input, args.input_text, "distilbert-base-cased-distilled-squad")

  benchmark_test = partial(perf_test, iters_number = args.iters_number, model_name = "Distilbert-base-cased distilled by SQuAD")

  if(args.tvm):
    from tvm_utils import tvm_test
    tvm_test(benchmark_test, onnx_model, encoded_inputs, opt_level, args.target, args.target, freeze)

  if args.ort:
    from ort_utils import ort_test
    ort_test(benchmark_test, args.model_path, encoded_inputs)
