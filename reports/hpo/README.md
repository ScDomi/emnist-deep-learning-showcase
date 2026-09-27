# Hyperparameter tuning summary

Optuna was used to tune learning rate, batch size, dropout and exponential learning-rate decay. The unusually specific final parameters are not random magic numbers; they are the best values returned by the search and then reused for the final training run.

| Model | Best trial | Best val accuracy | LR | Batch size | Dropout | LR decay |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 24 | 0.8362 | 0.0002355648 | 32 | 0.1000649470 | 0.9068206566 |
| Modular classifier | 7 | 0.8318 | 0.0003737883 | 128 | 0.2404434196 | 0.9836439484 |

## Visuals

![Best trial progress](../figures/hpo_best_trial_progress.png)

![Optuna trial search](../figures/hpo_trial_search.png)

![Parameter sensitivity](../figures/hpo_parameter_sensitivity.png)

## Generated files

- `reports/hpo/optuna_trial_summary.csv`
- `reports/hpo/optuna_epoch_history.csv`
- `reports/hpo/resnet18_best_params.json`
- `reports/hpo/modular_classifier_best_params.json`
- `reports/figures/hpo_best_trial_progress.png`
- `reports/figures/hpo_trial_search.png`
- `reports/figures/hpo_parameter_sensitivity.png`
