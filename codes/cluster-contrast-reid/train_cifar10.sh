CUDA_VISIBLE_DEVICES=0 python examples/cluster_contrast_train_usl.py -b 60 -a alexnet -d imagenet --hash-bit 16 --iters 400 --momentum 0.1 --eps 0.6 --num-instances 3 --pooling-type gem --use-hard 
