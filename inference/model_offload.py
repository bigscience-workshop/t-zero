# coding=utf-8

"""
Script showcasing how to run inference of T0++ on a single GPU using offloading.
It relies on Deepspeed (https://github.com/microsoft/DeepSpeed) and the ZeRO-3 offloading implementation.

The script is adapted from https://huggingface.co/transformers/main_classes/deepspeed.html#non-trainer-deepspeed-integration
"""


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid warnings about parallelism in tokenizers

model_name = "bigscience/T0pp"

ds_config = {
    "fp16": {
        "enabled": False,
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "stage3_param_persistence_threshold": 4e7, # Tune this value depending on the capacity of your GPU. With the current value, the GPU memory will peak at ~24GB.
    },
    "train_batch_size": 1,
}

_ = HfDeepSpeedConfig(ds_config)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded")

inputs = tokenizer.encode("Review: this is the best cast iron skillet you will ever buy. Is this review positive or negative?", return_tensors="pt")
inputs = inputs.to("cuda:0")

deepspeed_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config_params=ds_config,
    model_parameters=None,
    optimizer=None,
    lr_scheduler=None
)

deepspeed_engine.module.eval()
with torch.no_grad():
    outputs = deepspeed_engine.module.generate(inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("FINISHED")
