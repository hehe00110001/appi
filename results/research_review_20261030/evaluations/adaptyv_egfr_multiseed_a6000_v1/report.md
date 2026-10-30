# EGFR external evaluation

Frozen checkpoints, no EGFR fitting/calibration. Round2 is the primary structural cohort. Unknown/unexpressed omitted from binding classification; missing KD never imputed. Quantitative primary excludes unresolved 10uM boundary values. This is a known/homologous-target external design test.

|Model/checkpoint|Predicted/total|Binding n (+)|AUROC|Average precision|KD n|pKd MAE|pKd Pearson|pKd Spearman|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|structure_transfer_seed42/best|400/400|378 (53)|0.496|0.135|44|1.268|-0.067|-0.052|
|structure_transfer_seed42/best_absolute|400/400|378 (53)|0.542|0.150|44|1.073|-0.241|-0.271|
|fusion_absolute_seed42/best|400/400|378 (53)|0.534|0.149|44|1.104|-0.272|-0.344|
|fusion_absolute_seed42/best_absolute|400/400|378 (53)|0.480|0.134|44|0.856|-0.152|-0.159|
|fusion_transfer_seed42/best|400/400|378 (53)|0.486|0.142|44|1.484|0.110|0.128|
|fusion_transfer_seed42/best_absolute|400/400|378 (53)|0.554|0.154|44|1.539|-0.204|-0.205|
|fusion_shuffled_seed42/best|400/400|378 (53)|0.600|0.187|44|1.647|-0.182|-0.271|
|fusion_shuffled_seed42/best_absolute|400/400|378 (53)|0.600|0.187|44|1.647|-0.182|-0.271|
|historical_seq_B_seed42|400/400|378 (53)|0.317|0.101|44|1.169|-0.089|-0.083|
|structure_transfer_seed43/best|400/400|378 (53)|0.501|0.169|44|1.302|0.088|-0.049|
|structure_transfer_seed43/best_absolute|400/400|378 (53)|0.628|0.189|44|0.933|-0.119|-0.155|
|fusion_absolute_seed43/best|400/400|378 (53)|0.547|0.149|44|0.797|0.241|0.229|
|fusion_absolute_seed43/best_absolute|400/400|378 (53)|0.449|0.131|44|0.696|0.399|0.401|
|fusion_transfer_seed43/best|400/400|378 (53)|0.575|0.192|44|1.324|-0.011|-0.163|
|fusion_transfer_seed43/best_absolute|400/400|378 (53)|0.619|0.188|44|0.915|-0.243|-0.195|
|fusion_shuffled_seed43/best|400/400|378 (53)|0.582|0.179|44|2.039|0.037|-0.028|
|fusion_shuffled_seed43/best_absolute|400/400|378 (53)|0.670|0.209|44|1.265|-0.038|-0.193|
|structure_transfer_seed44/best|400/400|378 (53)|0.641|0.185|44|1.826|0.133|0.125|
|structure_transfer_seed44/best_absolute|400/400|378 (53)|0.615|0.170|44|1.322|-0.017|-0.029|
|fusion_absolute_seed44/best|400/400|378 (53)|0.461|0.130|44|0.980|-0.057|-0.089|
|fusion_absolute_seed44/best_absolute|400/400|378 (53)|0.451|0.130|44|0.906|-0.009|0.123|
|fusion_transfer_seed44/best|400/400|378 (53)|0.671|0.201|44|1.677|0.231|0.276|
|fusion_transfer_seed44/best_absolute|400/400|378 (53)|0.569|0.165|44|1.303|0.240|0.303|
|fusion_shuffled_seed44/best|400/400|378 (53)|0.517|0.141|44|2.456|-0.326|-0.294|
|fusion_shuffled_seed44/best_absolute|400/400|378 (53)|0.557|0.181|44|0.601|0.414|0.246|

Sequence-cluster intervals are in cluster_intervals.json; strata and boundary sensitivity in metrics.json. Round1 uses a different docking pipeline and is reported separately in metrics.json. Primary and secondary checkpoints stay separate; neither is chosen by external performance.
