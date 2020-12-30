python bert.py \
    --epochs=80 \
    --batch_size=2 \
    --lr=1e-5 \
    --max_length=512 \
    --pretrained_model=hfl/chinese-roberta-wwm-ext-large \
    --save_dir=bert-based-model/2020-12-13-00:43.bin \
    --gpu_id=0 \
    --labels_list=../data/labels.txt 