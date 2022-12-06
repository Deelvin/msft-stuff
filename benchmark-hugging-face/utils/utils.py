import time
import numpy as np

from transformers import DistilBertTokenizer


DISTILBERT_TEST_TEXT = """
Replace me by any text you'd like. DistilBERT is a distilled version of the BERT mode with 40 percent smaller size 
and 60 percent faster inference time, while preserving over 95 percent of BERT's accuracy on the GLUE language understanding task.
"""

def perf_test(run, iters_number = 1000, model_name = "ResNet50-v1", framework_name = "TVM+VM"):
  assert iters_number > 0

  tic = time.perf_counter()
  for i in range(iters_number):
      run()
  toc = time.perf_counter()
  print(f"Elapsed time: {toc - tic:0.4f} seconds for {iters_number} iterations of inference of {model_name} model by {framework_name}")

def get_onnx_input_name(model):
  inputs = [node.name for node in model.graph.input]
  initializer = [node.name for node in model.graph.initializer]

  inputs = list(set(inputs) - set(initializer))
  return inputs

def get_distilbert_inputs(artificial: bool,
                          input_text : str = DISTILBERT_TEST_TEXT,
                          tag : str = "distilbert-base-uncased"):
  inputs = {}
  if artificial:
    shape = [1, 128]
    inputs = {
      "input_ids": np.random.uniform(size=shape, low = 0, high=2000).astype("int64"),
      "attention_mask": np.ones(shape).astype("int64"),
    }
  else:
    print("Input text: ", input_text)
    tokenizer = DistilBertTokenizer.from_pretrained(tag)
    inputs = dict(tokenizer(input_text, return_tensors='np'))
  return inputs
