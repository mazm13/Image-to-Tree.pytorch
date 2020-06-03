python scripts/prepro_tree_labels.py \
    --input_json dataset/raw_coco_tree.json \
    --output_json data/cocotree.json \
    --output_h5 data/cocotree \

python scripts/prepro_valtest_labels.py \
    --input_json data/dataset_coco.json \
    --input_tree_json data/cocotree.json \
    --output_json data/cocotalk.json \
    --output_h5 data/cocotalk 