```shell script
# acgan without gradients penalty
python train.py --model acgan --latent_dim 168 --label_dim 3 --batch_size 32 --epoch 101 --soft_gpu -lr 0.0002 --beta1 0.5 --beta2 0.9 --net resnet

# acgan with gradients penalty
python train.py --model acgangp --latent_dim 168 --label_dim 3 --batch_size 32 --epoch 101 --soft_gpu --lambda 10 --d_loop 1 -lr 0.0001 --beta1 0 --beta2 0.9 --net dcnet
python train.py --model acgangp --latent_dim 168 --label_dim 3 --batch_size 32 --epoch 101 --soft_gpu --lambda 10 --d_loop 2 -lr 0.0001 --beta1 0. --beta2 0.9 --net resnet 
```

```shell script
tensorborad --logdir visual/acgan
tensorborad --logdir visual/acgangp
```

## acgan
![](demo/acgan-ep-200.png)

## acgan with gp
![](demo/ep-190.png)
