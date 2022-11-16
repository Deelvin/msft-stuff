import argparse
from functools import partial

import onnx
from transformers import DistilBertTokenizer

import tvm
from tvm import relay

from utils import perf_test, get_tvm_vm


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
  parser.add_argument("-t", "--target", default="llvm -mcpu=skylake-avx512", type=str, help=\
    "Target for model inference")
  parser.add_argument("-n", "--iters_number", default=1000, type=int, help=\
    "Number of iterations of inference for performance measurement")

  args = parser.parse_args()

  print("Input text: ", args.input_text)

  target = args.target
  target_host = target
  opt_level = 3
  freeze = True

  onnx_model = onnx.load(args.model_path)

  pretrained_weights = 'distilbert-base-cased-distilled-squad'

  tokenizer = DistilBertTokenizer.from_pretrained(pretrained_weights)
  encoded_input = tokenizer(args.input_question, args.input_text, return_tensors='np')

  benchmark_test = partial(perf_test, iters_number = args.iters_number, model_name = "Distilbert-base-cased distilled by SQuAD")

  if(args.tvm):
    print("----- TVM testing -----")
    shape_dict = {input_name: input.shape for (input_name, input) in encoded_input}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    mod = relay.transform.DynamicToStatic()(mod)

    tvm_inputs = {input_name: tvm.nd.array(input) for (input_name, input) in encoded_input}

    dev = tvm.device(str(target), 0)
 
    m = get_tvm_vm(mod, opt_level, target, target_host, params, dev)
    m.set_input(**tvm_inputs)
    benchmark_test(m.run)
