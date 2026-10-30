APPI seed44 is a research candidate for scoring protein–protein complexes from frozen PPIformer and full-chain ESM-2 features. ESM context routes structural attention; raw structural features remain available to the shared pKd readout. Experimental mutation differences supervise the same score used for absolute affinity.

This release packages `fusion_transfer_seed44/joint/best.pt`, joint epoch 16, as inference-only weights with normalization buffers, configuration, checksums and a cached-feature prediction CLI. Third-party encoder weights and feature caches are not included. This is not a raw-PDB-to-affinity service.

Seed44 was selected for release after inspecting internal and EGFR results. Its EGFR Round2 binding AUROC is 0.672 (378 known-status designs); quantitative Pearson is 0.231 and MAE is 1.677 pKd (44 measured affinities). The three-seed AUROCs are 0.486, 0.575 and 0.672. High-affinity top-k enrichment has not been demonstrated. Complete comparisons and limitations accompany the model card and result snapshot.

Use this prerelease for research reproduction and further model development. It does not establish reliable absolute affinity prediction or unseen-target generalization.
