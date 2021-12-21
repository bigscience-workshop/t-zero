# coding=utf-8

"""
Script showcasing how to run inference of T0++ on multiple GPUs using model parallelism. The model will be splitted across all available devices.
Note that this feature is still an experimental feature under 🤗 Transformers.

The minimum requirements to run T0++ (11B parameters) inference are 4 16GB V100 or 2 32GB V100 (or basically, enough GPU memory to fit ~42GB of fp32 parameters).
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "bigscience/T0pp"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded")

model.parallelize()
print("Moved model to GPUs")

inputs = tokenizer.encode("Review: this is the best cast iron skillet you will ever buy. Is this review positive or negative?", return_tensors="pt")
inputs = inputs.to("cuda:0")
with torch.no_grad():
    outputs = model.generate(inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("FINISHED")
