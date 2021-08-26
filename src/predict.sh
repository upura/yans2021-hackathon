export CUDA_VISIBLE_DEVICES="0,1"
python predict.py \
    --input_path /path/to/Target_Category \
    --model_path /path/to/model_file \
    --mode all
