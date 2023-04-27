for S in 0
do
    python myol.py --dataset cifar100 --batch_size 128 --mixup --ablation 0 --seed $S
    python myol.py --dataset cifar100 --batch_size 128 --mixup --ablation 1 --seed $S
    python myol.py --dataset cifar100 --batch_size 128 --mixup --ablation 2 --seed $S
    python myol.py --dataset cifar100 --batch_size 128 --mixup --ablation 3 --seed $S
    python simclr-eval.py --dataset cifar100 --model_name myol_ablation0 --checkpoint 100 --seed $S
    python simclr-eval.py --dataset cifar100 --model_name myol_ablation1 --checkpoint 100 --seed $S
    python simclr-eval.py --dataset cifar100 --model_name myol_ablation2 --checkpoint 100 --seed $S
    python simclr-eval.py --dataset cifar100 --model_name myol_ablation3 --checkpoint 100 --seed $S
done
for S in 0 1 2
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
# noise
# for S in 0 1 2
# do
#     for N in 0.05 0.01 0.1 0.005 0.001
#     do
#         python myol.py --batch_size 128 --mixup --ablation 0 --seed $S --noise $N
#         python simclr-eval.py --model_name myol_ablation0_noise"$N" --checkpoint 100 --seed $S
#     done
# done