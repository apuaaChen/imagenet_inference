python main.py /raid/datasets/ImageNet2012/ -j=8 -a=$1 --pretrained --epochs=3 -b=$2 --gpus=1 --mode=inference --log=$3