# Frozen EGFR evaluation across seeds

Mean and sample SD across seeds, not independent test cohorts. No seed or checkpoint selected on EGFR. Historical sequence reference remains seed42 only.

Round2/all is primary. Round1, strata and 10uM sensitivity remain separate in metrics.json.

## best

|Arm|AUROC|AP|pKd MAE|Pearson|Spearman|
|---|---:|---:|---:|---:|---:|
|structure_transfer|0.546 ± 0.082|0.163 ± 0.025|1.465 ± 0.313|0.051 ± 0.105|0.008 ± 0.101|
|fusion_absolute|0.514 ± 0.046|0.143 ± 0.011|0.960 ± 0.155|-0.029 ± 0.257|-0.068 ± 0.287|
|fusion_transfer|0.577 ± 0.093|0.178 ± 0.032|1.495 ± 0.177|0.110 ± 0.121|0.081 ± 0.223|
|fusion_shuffled|0.566 ± 0.044|0.169 ± 0.024|2.047 ± 0.405|-0.157 ± 0.183|-0.198 ± 0.147|

## best_absolute

|Arm|AUROC|AP|pKd MAE|Pearson|Spearman|
|---|---:|---:|---:|---:|---:|
|structure_transfer|0.595 ± 0.046|0.170 ± 0.020|1.109 ± 0.197|-0.125 ± 0.112|-0.152 ± 0.121|
|fusion_absolute|0.460 ± 0.017|0.132 ± 0.002|0.819 ± 0.110|0.080 ± 0.286|0.122 ± 0.280|
|fusion_transfer|0.581 ± 0.034|0.169 ± 0.017|1.252 ± 0.315|-0.069 ± 0.268|-0.033 ± 0.290|
|fusion_shuffled|0.609 ± 0.057|0.192 ± 0.015|1.171 ± 0.530|0.064 ± 0.311|-0.073 ± 0.279|

