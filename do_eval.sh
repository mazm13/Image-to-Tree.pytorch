python eval.py \
    --model log_tree_att/model-best.pth \
    --infos_path log_tree_att/infos_tree_att-best.pkl \
    --only_lang_eval 1 \
    --force 1 \
    --input_tree_json data/cocotree.json \
    --input_json data/cocotalk.json \
    --input_fc_dir data/cocotalk_fc \
    --input_att_dir data/cocotalk_att \
    --input_label_h5 data/cocotalk_label.h5 \
    --input_treelabel_h5 data/cocotree_label.h5 \
    --batch_size 10 \
    --beam_size 1
