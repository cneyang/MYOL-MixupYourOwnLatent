for S in 2
do
    python myol.py --dataset cifar100 --batch_size 128 --mixup --ablation 0 --seed $S
    python myol.py --dataset cifar100 --batch_size 128 --mixup --ablation 01 --seed $S
    python myol.py --dataset cifar100 --batch_size 128 --mixup --ablation 02 --seed $S
    python myol.py --dataset cifar100 --batch_size 128 --mixup --ablation 03 --seed $S
    python myol.py --dataset cifar100 --batch_size 128 --mixup --ablation 012 --seed $S
    python myol.py --dataset cifar100 --batch_size 128 --mixup --ablation 0123 --seed $S
    python simclr-eval.py --dataset cifar100 --model_name myol_ablation0 --checkpoint 100 --seed $S
    python simclr-eval.py --dataset cifar100 --model_name myol_ablation01 --checkpoint 100 --seed $S
    python simclr-eval.py --dataset cifar100 --model_name myol_ablation02 --checkpoint 100 --seed $S
    python simclr-eval.py --dataset cifar100 --model_name myol_ablation03 --checkpoint 100 --seed $S
    python simclr-eval.py --dataset cifar100 --model_name myol_ablation012 --checkpoint 100 --seed $S
    python simclr-eval.py --dataset cifar100 --model_name myol_ablation0123 --checkpoint 100 --seed $S
done