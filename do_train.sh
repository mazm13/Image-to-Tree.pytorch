export CUDA_VISIBLE_DEVICES=0
model_id=tree_att
python train.py --id ${model_id} \
    --caption_model tree_att \
    --input_tree_json data/cocotree.json \
    --input_json data/cocotalk.json \
    --input_fc_dir data/cocotalk_fc \
    --input_att_dir data/cocotalk_att \
    --input_label_h5 data/cocotalk_label.h5 \
    --input_treelabel_h5 data/cocotree_label.h5 \
    --batch_size 40 \
    --learning_rate 1.25e-4 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log_${model_id} \
    --save_history_ckpt 1 \
    --save_checkpoint_every 6000 \
    --val_images_use 3200 \
    --max_epochs 30 \
    --language_eval 1 
