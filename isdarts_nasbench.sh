echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for data path, benchmark file path and log name"
  exit 1
fi

xpath=$1
bench_path=$2
log=$3

nohup python ISDARTS.py \
	--data_path ${xpath} \
	--arch_nas_dataset ${bench_path} \
	--config_path configs/nas-benchmark/algos/DARTS.config \
	--search_space_name nas-bench-201 \
	--model_config configs/search-archs/DARTS-NASBENCH.config \
	--arch_nas_dataset NAS-Bench-201-v1_0-e61699.pth \
	--shrink_steps 4 \
	--shrink_intervals 2 \
	>${log}.log 2>&1 &
