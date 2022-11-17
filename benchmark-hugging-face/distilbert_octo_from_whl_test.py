import argparse
from functools import partial

import distilbert
from transformers import DistilBertTokenizer

from utils import DISTILBERT_TEST_TEXT, perf_test


if __name__ == "__main__":
  class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

  parser = argparse.ArgumentParser(
    description="Performance test of Distilbert-base-uncased model (hugging face model zoo: " +
                "https://huggingface.co/distilbert-base-uncased/tree/main) by OctoML tools " +
                "(instructions: https://app.octoml.ai/model-hub/natural-language-processing/models/3f9b3a3d-ef06-4cc2-bb66-aefed8e4afd5/card)",
    formatter_class=MyFormatter
  )
  # Model format
  parser.add_argument("-i", "--input_text", default=DISTILBERT_TEST_TEXT, type=str, help=\
    "Test input text for Distilbert-base-uncased model")
  parser.add_argument("-n", "--iters_number", default=1000, type=int, help=\
    "Number of iterations of inference for performance measurement")

  args = parser.parse_args()

  benchmark_test = partial(perf_test, iters_number = args.iters_number, model_name = "Distilbert-by-OctoML")
  best_model = distilbert.OctomizedModel()

  tz = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
  inputs = tz(args.input_text, return_tensors='np')

  octo_runner = partial(best_model.run, *inputs.values())

  benchmark_test(octo_runner, framework_name = "OctoML")
