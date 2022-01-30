# T-Zero

This repository serves primarily as codebase and instructions for training, evaluation and inference of T0.

T0 is the model developed in [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207). In this paper, we demonstrate that massive multitask prompted fine-tuning is extremely effective to obtain task zero-shot generalization. T0 outperforms or matches GPT-3 while being 16x smaller.

While the codebase in this repository mainly reproduces and replicates the training and evaluation of T0, it will be useful for future research on massively multitask fine-tuning.

## Setup

1. Download the repo
2. Navigate to root directory of the repo
3. Run `pip install -e .` to install the `t0` module. Depending on your application you can run multiple flavors:
   1. `seqio_tasks`: Provide original seqio tasks used for the massively multitask fine-tuning. You can run `pip install -e .[seqio_tasks]` to install the extra requirements.

## Contents

- [Training](training/README.md): reproducing (or replicating) the massively multitask fine-tuning
- [Evaluation](evaluation/README.md): reproducing the main results reported in the paper
- [Inference](inference/README.md): running inference with T0
- [Examples](examples/README.md): fine-tuning T0 with additional datasets or prompts.

## Released checkpoints

Below are the links to the models reported in our paper. We recommend using [the T0++ checkpoint](https://huggingface.co/bigscience/T0pp) as it yields the best performance on the most tasks. Meanwhile, the [T0](https://huggingface.co/bigscience/T0) and [T0+](https://huggingface.co/bigscience/T0p) checkpoints are intended for zero-shot evaluations on held-out tasks. See Sections 3 and 5 of our paper for more details.

If you don’t have enough resources to run T0, a smaller version with 3 billion parameters ([T0 3B](https://huggingface.co/bigscience/T0_3B)) is also available. Note that it is trained with the same mixture of datasets as T0 (not T0++).

Lastly, if you want to study the effect of multitask prompted training (a.k.a. instruction tuning) itself, the checkpoints from our ablation studies may be helpful. [T0 Single Prompt](https://huggingface.co/bigscience/T0_single_prompt) trains on one prompt per dataset, while [T0 Original Task Only](https://huggingface.co/bigscience/T0_original_task_only) trains on an average of 5.7 prompts per datasets (cf. T0 vanilla trains on 8.03 prompts per dataset). Using this series of checkpoints allows you to measure, for example, as you increase the number of prompts per dataset, how the performance on some held-out X increases/decreases or behavior on a linguistic diagnostic set changes. See Section 6.2 of our paper for more details.

- T-Zero: https://huggingface.co/bigscience/T0
- T-Zero +: https://huggingface.co/bigscience/T0p
- T-Zero ++: https://huggingface.co/bigscience/T0pp
- T-Zero Single Prompt: https://huggingface.co/bigscience/T0_single_prompt
- T-Zero Original Task Only: https://huggingface.co/bigscience/T0_original_task_only
- T-Zero 3B: https://huggingface.co/bigscience/T0_3B

## Citation

If you find this resource useful, please cite the paper introducing T0:

```bibtex
@inproceedings{sanh2022multitask,
    title={Multitask Prompted Training Enables Zero-Shot Task Generalization},
    author={Victor Sanh and Albert Webson and Colin Raffel and Stephen Bach and Lintang Sutawika and Zaid Alyafeai and Antoine Chaffin and Arnaud Stiegler and Arun Raja and Manan Dey and M Saiful Bari and Canwen Xu and Urmish Thakker and Shanya Sharma Sharma and Eliza Szczechla and Taewoon Kim and Gunjan Chhablani and Nihal Nayak and Debajyoti Datta and Jonathan Chang and Mike Tian-Jian Jiang and Han Wang and Matteo Manica and Sheng Shen and Zheng Xin Yong and Harshit Pandey and Rachel Bawden and Thomas Wang and Trishala Neeraj and Jos Rozen and Abheesht Sharma and Andrea Santilli and Thibault Fevry and Jason Alan Fries and Ryan Teehan and Teven Le Scao and Stella Biderman and Leo Gao and Thomas Wolf and Alexander M Rush},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=9Vrb9D0WI4}
}
```
