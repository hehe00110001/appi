# Structure-context pilot

Primary: validation macro order, tie difference MAE; secondary: validation absolute MAE. Test and external16 never select.

One-seed results are exploratory. Same optimizer-update budget does not imply same task exposures or compute.

|Run|Absolute test MAE|Difference MAE|Zero-change MAE|Macro order|≥5, excluding 1EMV|
|---|---:|---:|---:|---:|---:|
|fusion_absolute_seed42|2.005|0.804|0.821|49.4%|48.0%|
|fusion_absolute_seed43|2.029|0.829|0.821|41.0%|46.7%|
|fusion_absolute_seed44|1.988|0.795|0.821|50.2%|47.1%|
|fusion_shuffled_seed42|2.355|0.785|0.821|49.1%|49.2%|
|fusion_shuffled_seed43|1.897|0.707|0.821|48.6%|52.6%|
|fusion_shuffled_seed44|2.461|0.740|0.821|50.7%|59.7%|
|fusion_transfer_seed42|2.041|0.671|0.821|57.6%|64.0%|
|fusion_transfer_seed43|2.452|0.711|0.821|54.5%|62.0%|
|fusion_transfer_seed44|1.646|0.676|0.821|49.6%|53.9%|
|structure_transfer_seed42|2.532|0.769|0.821|56.8%|60.2%|
|structure_transfer_seed43|2.190|0.700|0.821|57.0%|60.8%|
|structure_transfer_seed44|2.115|0.709|0.821|62.4%|58.3%|
