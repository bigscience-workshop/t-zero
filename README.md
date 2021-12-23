# T-Zero

This repository serves primarily as codebase and instructions for training, evaluation and inference of T0.

T0 is the model developed in [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207). In this paper, we demonstrate that massive multitask prompted fine-tuning is extremely effective to obtain task zero-shot generalization. T0 outperforms or matches GPT-3 while being 16x smaller.

While the codebase in this repository mainly reproduces and replicates the training and evaluation of T0, it will be useful for future research on massively multitask fine-tuning.

## Contents

- [Training](training/README.md): reproducing (or replicating) the massively multitask fine-tuning
- [Evaluation](evaluation/README.md): reproducing the main results reported in the paper
- [Inference](inference/README.md): running inference with T0

## Citation

If you find this resource useful, please cite the paper introducing T0:

```bibtex
@misc{sanh2021multitask,
      title={Multitask Prompted Training Enables Zero-Shot Task Generalization},
      author={Victor Sanh and Albert Webson and Colin Raffel and Stephen H. Bach and Lintang Sutawika and Zaid Alyafeai and Antoine Chaffin and Arnaud Stiegler and Teven Le Scao and Arun Raja and Manan Dey and M Saiful Bari and Canwen Xu and Urmish Thakker and Shanya Sharma Sharma and Eliza Szczechla and Taewoon Kim and Gunjan Chhablani and Nihal Nayak and Debajyoti Datta and Jonathan Chang and Mike Tian-Jian Jiang and Han Wang and Matteo Manica and Sheng Shen and Zheng Xin Yong and Harshit Pandey and Rachel Bawden and Thomas Wang and Trishala Neeraj and Jos Rozen and Abheesht Sharma and Andrea Santilli and Thibault Fevry and Jason Alan Fries and Ryan Teehan and Stella Biderman and Leo Gao and Tali Bers and Thomas Wolf and Alexander M. Rush},
      year={2021},
      eprint={2110.08207},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
