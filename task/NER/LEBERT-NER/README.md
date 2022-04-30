## Task 1：Medical Named Entity Recognition (Medical NER)

This dir contains the code of **LEBERT** model for the Medical NER task. 

The code is adapted from https://github.com/yangjianxin1/LEBERT-NER-Chinese.

### Requirements

- python==3.6
- transformers==3.1.0
- torch==1.10.0

```shell
pip install -r requirements.txt
```

### Preprocess 

```shell
python preprocess.py
```

### Training / Inference / Evaluation

Before training, please download word vector from [tencent-ailab-embedding-zh-d200-v0.2.0-s](https://ai.tencent.com/ailab/nlp/en/download.html), which contains two million pre-trained word vectors with a dimension of 200.

**Note**: The code uses a single script for training, inference, and evaluation, and decoupling the process requires some modifications.

```shell
DEVICE=1
DATA_SET='msra'
MODEL_CLASS='lebert-softmax'
LR=1e-5
CRF_LR=1e-3
ADAPTER_LR=1e-3
PRETRAIN_MODEL='bert-base-chinese'
export CUDA_VISIBLE_DEVICES=${DEVICE}

python train.py \
    --device gpu \
    --output_path output \
    --add_layer 1 \
    --loss_type ce \
    --lr ${LR} \
    --crf_lr ${CRF_LR} \
    --adapter_lr ${ADAPTER_LR} \
    --weight_decay 0.01 \
    --eps 1.0e-08 \
    --epochs 5 \
    --batch_size_train 64 \
    --batch_size_eval 256 \
    --num_workers 0 \
    --eval_step 1000 \
    --max_seq_len 128 \
    --max_word_num  3 \
    --max_scan_num 3000000 \
    --data_path datasets/${DATA_SET}/ \
    --dataset_name ${DATA_SET} \
    --model_class ${MODEL_CLASS} \
    --pretrain_model_path ${PRETRAIN_MODEL} \
    --pretrain_embed_path pretrain_model/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt \
    --seed 42 \
    --markup bios \
    --grad_acc_step 1 \
    --max_grad_norm 1.0 \
    --num_workers 0 \
    --warmup_proportion 0.1 \
    --load_word_embed \
    --do_train \
    --do_eval
```
