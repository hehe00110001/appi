# Frozen ESM2 controlled pilot

One seed (42); B=balanced absolute, R=B+0.3 difference+0.1 ranking. Selection: absolute validation RMSE. External16 is a previously inspected diagnostic, never a selection set.

|Run|Best epoch|Test RMSE|Test difference MAE|Macro group order accuracy|External16 r|IL17A r|IL23R r|
|---|---:|---:|---:|---:|---:|---:|---:|
|seq_B_seed42|18|2.358|0.823|60.8%|-0.165|-0.487|0.728|
|seq_R_seed42|18|2.314|0.845|44.3%|-0.246|-0.572|0.457|
|esm_graph_B_seed42|5|2.278|0.834|44.1%|-0.320|-0.769|0.023|
|esm_graph_R_seed42|5|2.077|0.822|46.3%|-0.276|-0.643|0.006|
|esm_nogeom_B_seed42|16|2.405|0.811|53.9%|-0.179|-0.345|0.274|
|esm_nogeom_R_seed42|16|2.471|0.818|44.3%|-0.196|0.192|-0.031|
|fusion_B_seed42|3|2.144|0.812|46.9%|-0.273|-0.911|0.199|
|fusion_R_seed42|3|2.165|0.804|45.7%|-0.254|-0.761|0.124|

Matched esm_graph vs esm_nogeom isolates distance input with the same interface crop. seq uses full CSV chains. Fusion uses 30% total modality dropout. Single-seed differences are exploratory; inspect >=5-variant groups, exclusion of 1EMV homology, clustered intervals, and target-specific external results before expansion.
