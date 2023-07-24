for D in tinyimagenet cifar10 cifar100 stl10
do
    for S in 0 1 2
    do
        python ablation.py --dataset $D --alpha 0.5 --seed $S
        python eval_ablation.py --dataset $D --alpha 0.5 --seed $S
        python ablation.py --dataset $D --alpha 2.0 --seed $S
        python eval_ablation.py --dataset $D --alpha 2.0 --seed $S
        # python ablation.py --dataset $D --gamma 0.1 --seed $S
        # python eval_ablation.py --dataset $D --gamma 0.1 --seed $S
        # python ablation.py --dataset $D --gamma 10.0 --seed $S
        # python eval_ablation.py --dataset $D --gamma 10.0 --seed $S
    done
done