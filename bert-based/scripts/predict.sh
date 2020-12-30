python prediction.py \
    --batch_size=2 \
    --max_length=512 \
    --pretrained_model=hfl/chinese-roberta-wwm-ext-large \
    --model_path=bert-based-model/2020-12-13-00:43.bin \
    --gpu_id=0 \
    --labels_list=../data/labels.txt \
    --test_file=../data/test_5.txt \
    --output_path=outputs-tsv/output-bert.tsv