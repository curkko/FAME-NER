SEEDS=(171 354 550 667 985)
N=1
K=1
mode=intra

for seed in ${SEEDS[@]}; do
    python3 main.py \
        --bert_model=bert-base-chinese \
        --gpu_device=1 \
        --seed=${seed} \
        --mode=${mode} \
        --N=${N} \
        --K=${K} \
        --similar_k=10 \
        --eval_every_meta_steps=100 \
        --name=10-k_100_2_32_3_max_loss_2_5_BIOES \
        --train_mode=span \
        --inner_steps=2 \
        --inner_size=8 \
        --max_ft_steps=30 \
        --lambda_max_loss=2 \
        --inner_lambda_max_loss=5 \
        --tagging_scheme=BIOES \
        --viterbi=hard \
        --concat_types=None \
        --max_meta_steps 500 \
        --ignore_eval_test

    python3 main.py \
        --bert_model=bert-base-chinese \
        --seed=${seed} \
        --gpu_device=1 \
        --lr_inner=1e-4 \
        --lr_meta=1e-4 \
        --mode=${mode} \
        --N=${N} \
        --K=${K} \
        --similar_k=10 \
        --inner_similar_k=10 \
        --eval_every_meta_steps=100 \
        --name=10-k_100_type_2_32_3_10_10 \
        --train_mode=type \
        --inner_steps=2 \
        --inner_size=8 \
        --max_ft_steps=30 \
        --concat_types=None \
        --lambda_max_loss=2.0 \
        --max_meta_steps 500 \
        --ignore_eval_test

    cp models-${N}-${K}-${mode}/bert-base-chinese-innerSteps_2-innerSize_8-lrInner_0.0001-lrMeta_0.0001-maxSteps_500-seed_${seed}-name_10-k_100_type_2_32_3_10_10/en_type_pytorch_model.bin models-${N}-${K}-${mode}/bert-base-chinese-innerSteps_2-innerSize_8-lrInner_3e-05-lrMeta_3e-05-maxSteps_500-seed_${seed}-name_10-k_100_2_32_3_max_loss_2_5_BIOES

    python3 main.py \
        --bert_model=bert-base-chinese \
        --gpu_device=1 \
        --max_meta_steps 500 \
        --seed=${seed} \
        --N=${N} \
        --K=${K} \
        --mode=${mode} \
        --similar_k=10 \
        --name=10-k_100_2_32_3_max_loss_2_5_BIOES \
        --concat_types=None \
        --test_only \
        --eval_mode=two-stage \
        --inner_steps=2 \
        --inner_size=8 \
        --max_ft_steps=30 \
        --max_type_ft_steps=3 \
        --lambda_max_loss=2.0 \
        --inner_lambda_max_loss=5.0 \
        --inner_similar_k=10 \
        --viterbi=hard \
        --tagging_scheme=BIOES
done

