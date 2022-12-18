# Tackling domain adaptation problem in regression by Meta-Distillation from Mixture-of-Experts (Work In Progress)
A course project & reproducibility challenge

### Introduction


### Algorithm


### Dataset Description


#### 1. `train.csv`
Data for training.

#### 2. `dev_in.csv`
Development data from the same domain in time and climate as of the `train.csv` data.

#### 3. `eval_in.csv`
Evaluation data from the same domain in time and climate as of the `train.csv` data.

#### 4. `dev_out.csv`
Data distributionally shifted in time and climate from `train.csv`.

#### 5. `eval_out.csv`
Data further distributionally shifted in climate and different time frame from `train.csv` and `dev_out.csv`. Can be configured to have overlap in climates with `dev_out.csv`. <br /><br />

If `no_meta == 'yes'`, a further 4 files will be generated:

#### 6. `dev_in_no_meta.csv`
Same as `dev_in.csv` with meta data (first 6 features including climate type) removed.

#### 7. `eval_in_no_meta.csv`
Same as `eval_in.csv` with meta data (first 6 features including climate type) removed.

#### 8. `dev_out_no_meta.csv`
Same as `dev_out.csv` with meta data (first 6 features including climate type) removed.

#### 9. `eval_out_no_meta.csv`
Same as `eval_out.csv` with meta data (first 6 features including climate type) removed.


### Results

### References 
[1] The original paper on the Meta-DMoE: Zhong, Tao, et al. "Meta-DMoE: Adapting to Domain Shift by Meta-Distillation from Mixture-of-Experts." arXiv preprint arXiv:2210.03885 (2022).

[2] Dataset used: Malinin, Andrey, et al. "Shifts: A dataset of real distributional shift across multiple large-scale tasks." arXiv preprint arXiv:2107.07455 (2021).

[3] Validation & metrics pipeline code borrowed from: https://github.com/Shifts-Project/shifts
