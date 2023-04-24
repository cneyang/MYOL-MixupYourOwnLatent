python myol.py --batch_size 128 --mixup --ablation 3 --seed $S
python myol.py --batch_size 128 --mixup --ablation 4 --seed $S
python simclr-eval.py --model_name myol_ablation3 --checkpoint 100 --seed $S
python simclr-eval.py --model_name myol_ablation4 --checkpoint 100 --seed $S
for S in 0 1 2
do
    python myol.py --batch_size 128 --mixup --ablation 0 --seed $S
    python myol.py --batch_size 128 --mixup --ablation 01 --seed $S
    python myol.py --batch_size 128 --mixup --ablation 02 --seed $S
    python myol.py --batch_size 128 --mixup --ablation 012 --seed $S
    python myol.py --batch_size 128 --mixup --ablation 0123 --seed $S
    python simclr-eval.py --model_name myol_ablation0 --checkpoint 100 --seed $S
    python simclr-eval.py --model_name myol_ablation01 --checkpoint 100 --seed $S
    python simclr-eval.py --model_name myol_ablation02 --checkpoint 100 --seed $S
    python simclr-eval.py --model_name myol_ablation012 --checkpoint 100 --seed $S
    python simclr-eval.py --model_name myol_ablation0123 --checkpoint 100 --seed $S
done