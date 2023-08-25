collection=charades
visual_feature=i3d_rgb_lgi
# root_path=data/

#CF
clip_scale_w=0.4 # 4
frame_scale_w=0.6 # 0.6
# eval
eval_context_bsz=500
eval_query_bsz=500

margin=0.2
intra_margin=0.2 # 0.2

# dynamic
exp_id=$1
device_ids=$2
root_path=$3

# training
python method_prvr/train.py \
--collection $collection \
--visual_feature $visual_feature \
--root_path $root_path \
--dset_name $collection \
--exp_id $exp_id \
--clip_scale_w $clip_scale_w \
--frame_scale_w $frame_scale_w \
--device_ids $device_ids \
--eval_context_bsz $eval_context_bsz \
--eval_query_bsz $eval_query_bsz \
--margin $margin \
--intra_margin $intra_margin \
--global_sample # uncomment to use