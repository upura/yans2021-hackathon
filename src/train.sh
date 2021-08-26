export CUDA_VISIBLE_DEVICES="0,1"
python train.py \
    --input_path /path/to/Target_Category \
    --save_path /path/to/save_directory \
    --additional_name "" \
    --lr 1e-5 \
    --bsz 32 \
    --epoch 50 \
    --grad_acc 1 \
    --grad_clip 1.0 \
