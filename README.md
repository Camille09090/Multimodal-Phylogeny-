## Overview

This repository provides the data, code, model weights, and phylogenetic tree files used in our study on morphology-guided phylogenetic reconstruction of Staphylinidae (Coleoptera: Staphylinidae).

The proposed framework integrates:

1. A genus-level morphological text bank curated from taxonomic literature.
2. Image–text alignment based on DINOv2 and LoRA fine-tuning.
3. Morphological feature extraction from beetle habitus images.
4. Bayesian phylogenetic reconstruction using RevBayes.

## Dataset

The beetle image dataset used in this study was obtained from the Rove-Tree-11 collection:
Hunt G., Pedersen J. (2023). Rove-Tree-11: A large-scale image dataset of Staphylinidae with taxonomic annotations and reference phylogeny.
Please obtain the original image dataset from the corresponding publication and repository.


## Morphological Text Bank

The file:
```text
text_data/genus_level_morphological_description.xlsx
```
contains standardized genus-level morphological descriptions for 44 genera of Staphylinidae.


## Training
The training framework combines:

* DINOv2 ViT-B/14 backbone
* LoRA adaptation
* Supervised Contrastive Learning
* Morphological text alignment


## Feature Extraction

To extract morphological image representations:
python scripts/extract_feature.py
The script generates CLS-token features from the fine-tuned DINOv2 model.


## Phylogenetic Reconstruction
Feature matrices can be converted to NEXUS format using:
python scripts/feature_to_nex.py

Phylogenetic inference is then performed using:
```bash
rb revbayes/build_tree.Rev
```

## Tree Files
The repository includes:
* Morphological_characteristics_tree.tree
  Morphology-derived phylogenetic tree inferred in this study.
* reference_phylogeny.nex
  Reference phylogeny from Rove-Tree-11 paper.


## Model Weights

The file:
weights/best_lora_proj.pth
contains the best-performing image–text alignment checkpoint obtained during training.

## License

This repository is released under the MIT License.
# Multimodal-Phylogeny-
