# Training

This section explains how to reproduce the T0 training: a massively multitask fine-tuning using TPUs and Mesh Tensorflow. These are the steps we used in the paper for training and evaluation.

We also release code to replicate this training in PyTorch (see [Replicating the training in PyTorch](#replicating-the-training-in0-PyTorch)).

Before starting, please make sure you have installed the dependencies listed in [`requirements.txt`](https://github.com/bigscience-workshop/t-zero/blob/master/requirements.txt).

## Data pre-processing

The first step is to pre-process and cache the data. We used [SeqIO](https://github.com/google/seqio), a library for processing sequential data to be fed into downstream sequence models. Essentially, SeqIO tokenizes (and caches) the input/target pairs and handles the training and evaluation mixtures.

You can pre-process a given task with the following command:

```bash
TASK=yelp_review_full_based_on_that
seqio_cache_tasks \
   --tasks=$TASK \
   --output_cache_dir=$MY_FAVORITE_OUTPUT_DIR \
   --module_import=promptsource.seqio_tasks
```

The full list of tasks in the mixture is obtained with the following command:

```python
import seqio
import promptsource.seqio_tasks

for task in seqio.MixtureRegistry.get("d4_gpt_sglue_train").tasks:
    print(task.name)
```

You'll likely be interested in the following mixtures:
- `d4_train`: training mixture for T0
- `d4_gpt_train`: training mixture for T0+
- `d4_gpt_sglue_train`: training mixture for T0++

For reproducibility, we have released an [already pre-processed version of the data](https://huggingface.co/datasets/bigscience/P3), which means you don't need to cache the data yourself. The only exception is [Story Cloze](https://cs.rochester.edu/nlp/rocstories/), which requires filling a form to get the data. Please refer to the previous SeqIO commands to cache the tasks related to Story Cloze once you have the dataset.

Please also note that the module `promptsource.seqio_tasks` will be merged into the current repository. This is [WIP](https://github.com/bigscience-workshop/t-zero/issues/6).

## Reproducing training in Mesh Tensorflow

*Please make sure you first install the [`t5` library](https://github.com/google-research/text-to-text-transfer-transformer) and its dependencies.*

Once you have pre-processed (and cached) the data, you can launch the training. We assume that you have access to TPU resources.

To launch the training, a typical command is (please adjust the arguments to fit your setup and training):
```bash
export TPU_NAME=your_tpu_name
export PROJECT=your_project_name
export ZONE=your_project_zone
export TPU_SIZE=v3-512

export BUCKET=gs://your_bucket/
export DATA_DIR="${BUCKET}/your_data_dir"
export MODEL_DIR="${BUCKET}/your_model_dir"

export MIXTURE_NAME="d4_gpt_sglue_train"
export TRAIN_STEPS=1112200

t5_mesh_transformer \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}"\
    --additional_task_cache_dirs="${DATA_DIR}" \
    --module_import="promptsource.seqio_tasks" \
    --gin_file="dataset.gin" \
    --gin_param="MIXTURE_NAME = '${MIXTURE_NAME}'" \
    --gin_file="gs://t5-data/pretrained_models/t5.1.1.lm100k.xxl/operative_config.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="run.train_steps = ${TRAIN_STEPS}" \
    --gin_file="learning_rate_schedules/constant_0_001.gin" \
    --gin_param="tokens_per_batch=1048576" \
    --gin_param="pack_dataset.use_custom_ops = False" \
    --gin_param="run.sequence_length = {'inputs': 1024, 'targets': 256}" \
    --gin_param="model_info = '${MODEL_DIR}/model-info.txt'" \
    --gin_param="mesh_train_dataset_fn.use_cached = True" \
    --gin_param="serialize_num_microbatches.tokens_per_microbatch_per_replica = 2048" \
    --gin_param="tpu_mesh_shape.model_parallelism = 8"
```

You can evaluate (using rank evaluation) the trained model with a similar command:

```bash
export TPU_NAME=your_tpu_name
export PROJECT=your_project_name
export ZONE=your_project_zone
export TPU_SIZE=v3-128

export BUCKET=gs://your_bucket/
export DATA_DIR="${BUCKET}/your_data_dir"
export MODEL_DIR="${BUCKET}/your_model_dir"

export EVAL_MIXTURE_NAME="d4_score_eval"
export TRAIN_STEPS=1112200

t5_mesh_transformer \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="${MODEL_DIR}/operative_config.gin" \
    --additional_task_cache_dirs="${DATA_DIR}" \
    --module_import="promptsource.seqio_tasks" \
    --gin_file="score_eval.gin" \
    --gin_param="MIXTURE_NAME = '${EVAL_MIXTURE_NAME}'" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="run.batch_size = ('tokens_per_batch', 262144)" \
    --gin_param="pack_dataset.use_custom_ops = False" \
    --gin_param="run.sequence_length = {'inputs': 1024, 'targets': 256}" \ --gin_param="model_info = '${MODEL_DIR}/model-info.txt'" \
    --gin_param="mesh_train_dataset_fn.use_cached = True" \
    --gin_param="mesh_eval_dataset_fn.use_cached = True" \
    --gin_param="serialize_num_microbatches.tokens_per_microbatch_per_replica = 2048" \
    --gin_param="tpu_mesh_shape.model_parallelism = 4" \
    --gin_param="eval_checkpoint_step = ${TRAIN_STEPS}" \
    --gin_param="run.dataset_split = 'validation'"
```

## Replicating the training in PyTorch

This section is still WIP.

To evaluate the trained models using PyTorch, please refer to [this section](https://github.com/bigscience-workshop/t-zero/tree/master/evaluation).
