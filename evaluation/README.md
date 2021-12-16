# Reproducing evaluation

To reproduce the main numbers (Figure 4) we reported in the paper:

```bash
python run_eval.py \
    --dataset_name super_glue \
    --dataset_config_name rte \
    --template_name "must be true" \
    --model_name_or_path bigscience/T0_3B \
    --output_dir ./debug
```

The list of templates per data(sub)set is available inside [this file](template_list.py).
