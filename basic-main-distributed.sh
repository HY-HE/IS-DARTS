echo script name: $0
echo $# arguments
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for dataset name, data path, log name and number of gpus"
  exit 1
fi

dataset=$1
xpath=$2
log=$3
n_gpus=$4

if [ ${dataset} == 'cifar10' ] || [ ${dataset} == 'cifar100' ]; then
  base=CIFAR
  cutout_length=16
  batch=96
  worker=4
elif [ ${dataset} == 'imagenet-1k' ]; then
  base=IMAGENET
  cutout_length=-1
  batch=128
  worker=16
else
  exit 1
  echo 'Unknown dataset: '${dataset}
fi

nohup python -m torch.distributed.launch --nproc_per_node ${n_gpus} basic-main-distributed.py --dataset ${dataset} \
	--data_path ${xpath} \
	--model_config ./configs/archs/NAS-${base}-ISDARTS.config \
	--optim_config ./configs/opts/NAS-${base}.config \
	--cutout_length ${cutout_length} \
	--batch_size ${batch} \
	--workers ${worker} \
	>${log}.log 2>&1 &
