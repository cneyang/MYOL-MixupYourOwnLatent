## Pretraining
```bash
# Pretraining
python train.py --dataset DATASET_NAME --algo ALGORITHM_NAME --epochs EPOCHS

# Linear Evaluation
python eval.py --dataset DATASET_NAME --algo ALGORITHM_NAME --checkpoint EPOCHS
```
### List of datasets and algorithms
- DATASET_NAME: `cifar10`, `cifar100`, `stl10`, `tinyimagenet`
- ALGORITHM_NAME: `simclr`, `moco`, `byol`, `tribyol`, `unmix`, `imix`, `myol`

## Downstream Tasks
```bash
# Semi-supervised learning
# RATIO: 0.1, 0.01
python semisup.py --labeled_ratio RATIO --dataset DATASET_NAME --algo ALGORITHM_NAME --checkpoint EPOCHS

# Transfer learning
# TARGET: cifar10, cifar100, stl10, mnist, fashionmnist, kmnist, usps, svhn
python tl.py --target TARGET --dataset DATASET_NAME --algo ALGORITHM_NAME --checkpoint EPOCHS

# Adversarial attack
python adversarial_attack.py --dataset DATASET_NAME --algo ALGORITHM_NAME --checkpoint EPOCHS
```

## Supplementary Material
See [Appendix.pdf](./Appendix.pdf) for full dataset results and pseudo-code.
