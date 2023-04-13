for S in 0 1
do
    for B in 128 64 32 16
    do
        python myol.py --dataset=cifar10 --batch_size=$B --seed=$S --mixup
        python myol.py --dataset=cifar10 --batch_size=$B --seed=$S
        for C in 100
        do
            python simclr-eval.py --dataset=cifar10 --batch_size=$B --algo=myol --model_name=myol --seed=$S --checkpoint=$C
            python simclr-eval.py --dataset=cifar10 --batch_size=$B --algo=byol --model_name=byol --seed=$S --checkpoint=$C
        done
    done
done