# Source Label Adaptation - MCL + SLA
CVPR 2023 SUBMISSION - Semi-Supervised Domain Adaptation with Source Label Adaptation

This code reproduces MCL and MCL + SLA.

## Requirement & Data Preparation
Following [MCL](https://github.com/chester256/MCL) to install related packages and prepare data.

## Running

1. MCL

Following [MCL](https://github.com/chester256/MCL) to reproduce MCL results.

2. MCL + SLA (take 3-shot A -> C OfficeHome dataset as an example.)

```
python train_mcl.py --method LC --alpha 0.3 --ppc_T 0.6 --update_interval 500 --warmup 5000 --source Art --target Clipart --seed 4158
```

## Acknowledgement
This code is almost the same as [the original one](https://github.com/chester256/MCL).

We only fix the `work_init_fn` and `generator` in the dataloader, and apply our SLA to it.
