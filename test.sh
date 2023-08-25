# root_path=data/

npev=10 # max for didemo
eval_query_bsz=200 # 500
eval_context_bsz=500
collection=$1
eval_id=$2
root_path=$3
model_dir=$4


if [ $collection == 'didemo' ]; then
    npev=40
fi

# training

python method/eval.py \
--collection $collection \
--eval_id $eval_id \
--root_path $root_path  \
--dset_name $collection \
--model_dir $model_dir \
--npev $npev \
--eval_context_bsz $eval_context_bsz \
--eval_query_bsz $eval_query_bsz