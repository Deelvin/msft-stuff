import argparse

import onnx
from transformers import DistilBertTokenizer

from tvm import relay

from utils import DISTILBERT_TEST_TEXT
from tvm_utils import tvm_tuning


if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Tuning of distilbert-like model by auto-scheduler from TVM",
        formatter_class=MyFormatter
    )
    # Model format
    parser.add_argument("-m", "--model_path", default="", type=str, help=\
        "The path to onnx-file with model")
    parser.add_argument("-n", "--trials_number", default=100, type=int, help=\
        "Number of iterations of inference for performance measurement")
    parser.add_argument("-t", "--target", default="llvm -mcpu=skylake-avx512", type=str, help=\
        "Target for model inference")

    args = parser.parse_args()

    onnx_model = onnx.load(args.model_path)
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    encoded_inputs = tokenizer(DISTILBERT_TEST_TEXT, return_tensors='np')

    print("----- TVM tuning -----")
    shape_dict = {input_name: input.shape for (input_name, input) in encoded_inputs.items()}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    log_file_name = args.model_path.replace(".onnx","") + "_tuned.json"

    # Model tuning by tvm autoscheduler
    tvm_tuning(
        mod,
        args.target,
        args.target,
        params,
        trials_num = args.trials_number,
        log_file = log_file_name,
    )
