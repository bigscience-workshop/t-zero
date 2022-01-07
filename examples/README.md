# Single Task Fine-Tuning (Optionally Few-Shot)
`single_task_fine_tune.py` fine-tunes T0 on a dataset with a specified promptsource template. Here is an example command:
```bash
python single_task_fine_tune.py \
    --dataset_name super_glue \
    --dataset_config_name rte \
    --template_name "does this imply" \
    --model_name_or_path bigscience/T0_3B \
    --output_dir ./debug \
    --parallelize
```

The per epoch evaluation results will be saved as a CSV file in the `output_dir`. By default, it trains on the whole dataset. Optionally, you can pass `--num_shots` to train it on a random subset of examples.

Like the zero-shot evaluation [script](../evaluation/run_eval.py), you are expected to provide `dataset_name`, `dataset_config_name`, and `template_name`. You can find the list of templates per dataset in [this file](../evaluation/template_list.py); these were the templates we used in the T0 paper. In `setup.py`, [`promptsource`](https://github.com/bigscience-workshop/promptsource) is pinned to v0.1.0, the version we used in the T0 paper to facilitate reproduction and replication. However, `promptsource` is being continously updated, so if you don't intend to reproduce the exact results from our paper, you may want to install the latest `promptsource` and call, for example, `DatasetTemplates("super_glue", "rte").all_template_names` to access the new templates.


## Distributed Training

Although still an experimental feature of 🤗 Transformers, the simplest way to train T0 is to pass the `--parallelize` flag as shown in the example above, which calls `model.parallize()` and splits the model over all visible GPUs.

To train T0 3B, you need at least around 48GB of GPU memory in theory, which in practice usually means at least two V100s (32GB version), three RTX3090s, or a single A6000. For T0 11B, you need at least eight V100s. (If you don't need training and only need inferencing, then the VRAM requirement is about 1/4 of training, i.e., a single 3090 for T0 3B, or a single A6000 for T0 11B.)

Of course, you can further reduce the VRAM requirement by using [DeepSpeed](https://huggingface.co/docs/transformers/main_classes/deepspeed) with [`accelerate`](https://github.com/huggingface/accelerate). Please refer to their documentation for installation and configuration.

## Miscellaneous Notes

1. If you just want to debug your code without running a large model taking up lots of resources, you can use `--model_name_or_path google/t5-base-lm-adapt`, which is the non-multitask-prompted-trained equivalent of T0 that is available in much smaller sizes.

2. T0 was trained and evaluated without adding special EOS tokens in its input sequences, which is the default in `single_task_fine_tune.py`. However, T5 was pretrained with EOS in its input sequences, and we have noticed that adding EOS (using the `--input_eos` flag) sometimes improves the zero-shot and few-shot performance of T0.

3. If you train or evaluate on ANLI , the `dataset_config_name` should be `train_r1`, `dev_r1`, `train_r2`,  `dev_r2`, etc., corresponding to its 3 rounds of adversarial filtering.

4. The estimated GPU memory requirement above assumes using the AdamW optimizer. Note that T0 was trained (in Mesh TensorFlow) with Adafactor, which is much more memory efficient. However, we found that using PyTorch's implementation of Adafactor converges nontrivially worse than AdamW. Of course, you're free to experiment with Adafactor. See the “additional training tips” section [here](https://huggingface.co/docs/transformers/master/en/model_doc/t5#training). If you have success with Adafactor, feel free to open an issue and we might update this readme.

5. Inferncing with fp16 yields bad results and is thus discouraged.
