CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 10 -a resnet_ibn50a -d cifar --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --pooling-type gem --use-hard 
