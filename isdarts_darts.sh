echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for dataset name, data path and log name"
  exit 1
fi

dataset=$1
xpath=$2
log=$3

if [ ${dataset} == 'cifar10' ] || [ ${dataset} == 'cifar100' ]; then
  base=CIFAR
  clip=5
elif [ ${dataset} == 'imagenet-1k' ]; then
  base=IMAGENET
  clip=-1
else
  exit 1
  echo 'Unknown dataset: '${dataset}
fi

nohup python ISDARTS.py \
  --dataset ${dataset} \
	--data_path ${xpath} \
	--gradient_clip ${clip} \
	--config_path configs/search-opts/DARTS-NASNet-${base}.config \
	--search_space_name darts \
	--model_config configs/search-archs/DARTS-NASNet.config \
	--shrink_steps 7 \
	--shrink_intervals 3 \
	--total_epochs 50 \
	>${log}.log 2>&1 &
