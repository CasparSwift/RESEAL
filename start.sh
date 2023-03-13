#!bin/bash

# GigaWord Train
# CUDA_VISIBLE_DEVICES='1' python3 main.py --pretrain_model=../bart-large/ --epoch_num=5 \
# 	--dataset=gigaword --test_batch_size=100 --batch_size=100 \
# 	--output_dir=/dataA/chenxiang/gigaword_0504/ \
# 	--max_src_length=128 --max_tgt_length=64

# GigaWord Test MSR
# CUDA_VISIBLE_DEVICES='0' python3 main.py --pretrain_model=../bart-large/ --epoch_num=1 \
# 	--dataset=gigaword --test_batch_size=100 --test --output_dir=/dataA/chenxiang/gigaword_0112/ \
# 	--giga_test_set=MSR --beam_size=4 --max_src_length=128 --max_tgt_length=100 \
# 	--tester=base_MSR_l2.0s20000 --ckpt_path=/dataA/chenxiang/gigaword_0112/bart_e1_s20000.pt --length_penalty=2.0 --early_stopping

# test DUC 2004 
# CUDA_VISIBLE_DEVICES=2 python3 main.py --pretrain_model=../bart-large/ --epoch_num=1 \
# 	--dataset=gigaword --test_batch_size=100 --test --output_dir=output \
# 	--giga_test_set=duc --beam_size=5 --max_src_length=128 --max_tgt_length=25 \
# 	--tester=base_duc --no_early_stopping --ckpt_path=/data1/chenxiang/gigaword_0112/bart_e1_s20000.pt

# 1127 English-EWT Train
# CUDA_VISIBLE_DEVICES='0' python3 main.py --pretrain_model=../bart-large/ \
# 	--dataset=english-ewt --batch_size=32 --epoch_num=10

# 1127 English-EWT Test DBA
# CUDA_VISIBLE_DEVICES='3' python3 main.py --test --pretrain_model=../bart-large/ \
# 	--dataset=english-ewt \
# 	--test_batch_size=64 --tester=DBA_b30 --logging_steps=10 --beam_size=30

# Train Parser
# CUDA_VISIBLE_DEVICES='0' python3 main.py \
# 	--dataset=depparse --batch_size=16 --epoch_num=20 --logging_steps=20 --max_src_length=180 \
# 	--learning_rate=0.01 --output_dir=output/depparse0/ --weight_decay=1e-8 \

# Test Parser
# CUDA_VISIBLE_DEVICES='0' python3 main.py --test \
# 	--dataset=depparse --test_batch_size=16 --epoch_num=15 --logging_steps=20 --max_src_length=180 \
# 	--output_dir=output/depparse1/ --tester=depparse \


# Train Dependency Prediction
# CUDA_VISIBLE_DEVICES='0' python3 main.py --pretrain_model=../bert-base-uncased \
# 	--dataset=Giga_Dep_Pred --output_dir=/dataA/chenxiang/kw_0108/ --epoch_num=10 \
# 	--logging_steps=500 --batch_size=200 --max_src_length=200 --max_word_length=140 --weight_decay=1e-5 \
# 	--tester=deppred --learning_rate=2e-5 \
# 	--giga_test_set=internal --random_seed=1024


# Test Base
# CUDA_VISIBLE_DEVICES='0' python3 main.py --test --pretrain_model=../bart-large/ \
# 	--dataset=english-ewt \
# 	--test_batch_size=8 --tester=base_b10 --logging_steps=10 --beam_size=10

# Train SR
# CUDA_VISIBLE_DEVICES='0,1' python3 main.py --pretrain_model=/data/pretrain/bart-large/ \
# 	--dataset=SR_en_ewt --logging_steps=100 --batch_size=16 --epoch_num=40 --output_dir=/data/chenxiang/SR_1209/ 

# Test SR
# CUDA_VISIBLE_DEVICES='0' python3 main.py --test --pretrain_model=/data/pretrain/bart-large/ \
# 	--dataset=SR_en_ewt --epoch_num=10 --output_dir=output/SR_1207 \
# 	--test_batch_size=1 --tester=SR_2 --logging_steps=10 --beam_size=40


CUDA_VISIBLE_DEVICES=2 python3 main.py --test --pretrain_model=../bart-large/ \
	--dataset=english-ewt --epoch_num=10 \
	--test_batch_size=10 --tester=dep_debug --logging_steps=10 \
	--beam_size=20 --prune_size=20 --parser_type=l2r --scorer=prob \
	--lamb=7.0 --rho=0.5 --alpha_func=1 --normalize=True --bank_count=dep --force_complete --reset_avail_states=True \


# CUDA_VISIBLE_DEVICES='2' python3 main.py --test --pretrain_model=../bart-large/ \
# 	--dataset=english-ewt --epoch_num=10 \
# 	--test_batch_size=64 --tester=base_b4_1 --logging_steps=10 \
# 	--beam_size=20

# CUDA_VISIBLE_DEVICES='3' python3 main.py --test --pretrain_model=../bart-large/ \
# 	--dataset=english-ewt --epoch_num=10 \
# 	--test_batch_size=64 --tester=DBA_b20 --logging_steps=10 \
# 	--beam_size=20



# WebNLG

# Train WebNLG 
# CUDA_VISIBLE_DEVICES=1 python3 main.py --pretrain_model=/dataB/pretrain/T5-base \
# 	--dataset=webnlg --output_dir=/dataA/chenxiang/webnlg/t5-base/ \
# 	--batch_size=20 --test_batch_size=20 --epoch_num=30 --learning_rate=5e-5 \
# 	--max_src_length=384 --max_tgt_length=384 --test_when_training --beam_size=4 --tester=base_b4 --random_seed=42

# Test WebNLG with existing checkpoints
# CUDA_VISIBLE_DEVICES=3 python3 main.py --test --pretrain_model=t5-base --epoch_num=10 \
# 	--dataset=webnlg --output_dir=/dataA/chenxiang/webnlg/t5-base/ \
# 	--beam_size=80 --tester=base_b80 --test_batch_size=4 \
# 	--max_src_length=384 --max_tgt_length=384

# Train WebNLG relation checker
# CUDA_VISIBLE_DEVICES=1 python3 main.py --pretrain_model=lstm --epoch_num=10 --learning_rate=1e-3 \
# 	--dataset=webnlg_rel --batch_size=100 --test_batch_size=100 --output_dir=/dataA/chenxiang/webnlg_rc_filt \
# 	--max_src_length=384 --tester=relpred --test_when_training

# Test WebNLG with dependency constrainted decoding
# CUDA_VISIBLE_DEVICES=2 python3 main.py --test --pretrain_model=/data2/pretrain/T5-base --epoch_num=10 \
# 	--dataset=webnlg --output_dir=/data1/chenxiang/webnlg/t5-base/ \
# 	--beam_size=80 --test_batch_size=1 \
# 	--max_src_length=384 --max_tgt_length=384 \
# 	--parser_type=rel_checker --scorer=webnlg_prob --lamb=0.5 --rho=0.5 --alpha_func=1 \
# 	--tester=dep-webnlg_b80 

# Train webedit with T5
# CUDA_VISIBLE_DEVICES=2 python3 main.py --pretrain_model=/dataA/pretrain/T5-base --epoch_num=10 \
# 	--dataset=webedit --test_batch_size=50 --batch_size=32 --output_dir=/dataA/chenxiang/edit \
# 	--max_src_length=400 --max_tgt_length=200

# Test webedit with T5
# CUDA_VISIBLE_DEVICES=2 python3 main.py --test --pretrain_model=/dataA/pretrain/T5-base \
# 	--dataset=webedit --test_batch_size=100 --batch_size=32 --ckpt_path=/dataA/chenxiang/edit/t5_webedit_e5.pt \
# 	--max_src_length=400 --max_tgt_length=200 --beam_size=4 --tester=base_b4 --early_stopping


# Train webedit with LSTM
# CUDA_VISIBLE_DEVICES=3 python3 main.py --pretrain_model=dual --epoch_num=30 --learning_rate=1e-3 \
# 	--dataset=webedit_lstm --test_batch_size=64 --batch_size=200 --output_dir=/dataA/chenxiang/webedit_lstm_dual_copy2 \
# 	--logging_steps=100 \

# Test webedit with LSTM
# CUDA_VISIBLE_DEVICES=0 python3 main.py --test --pretrain_model=gru --epoch_num=50 \
# 	--dataset=webedit_lstm --test_batch_size=100 --output_dir=/dataA/chenxiang/webedit_lstm_0415 \
# 	--beam_size=4 --early_stopping \
# 	--tester=base-lstm_b4 \

# CUDA_VISIBLE_DEVICES=2 python3 main.py --test --pretrain_model=dual --epoch_num=25 \
# 	--dataset=webedit_lstm --test_batch_size=100 --output_dir=/dataA/chenxiang/webedit_lstm_dual_copy2 \
# 	--max_src_length=128 --max_tgt_length=128 --beam_size=4 --early_stopping \
# 	--tester=base-lstm_b4 \

# Train webedit relation checker
# CUDA_VISIBLE_DEVICES=3 python3 main.py --pretrain_model=lstm --epoch_num=10 --learning_rate=1e-3 \
# 	--dataset=webedit_rel --batch_size=300 --test_batch_size=300 --output_dir=/dataA/chenxiang/webedit_rc_filt_aug \
# 	--max_src_length=128 --logging_steps=100 --tester=relpred --test_when_training

# Test webedit by dependency constrained
# CUDA_VISIBLE_DEVICES=0 python3 main.py --test --pretrain_model=dual --epoch_num=20 \
# 	--dataset=webedit_lstm --test_batch_size=5 --output_dir=/dataA/chenxiang/webedit_lstm_dual_copy2 \
# 	--max_src_length=128 --max_tgt_length=128 --beam_size=40 --tester=dep-lstm_b40lamb0.5 \
# 	--parser_type=rel_checker --scorer=lstm_prob --lamb=0.5 --rho=0.5 --early_stopping --alpha_func=1 --force_complete 

# Train rotoedit with LSTM
# CUDA_VISIBLE_DEVICES=3 python3 main.py --pretrain_model=dual --epoch_num=30 --learning_rate=2e-3 \
# 	--dataset=rotoedit_lstm --test_batch_size=15 --batch_size=15 --output_dir=/dataA/chenxiang/rotoedit_lstm_dual_copy

# Test rotoedit with LSTM
# CUDA_VISIBLE_DEVICES=2 python3 main.py --test --pretrain_model=dual --epoch_num=30 \
# 	--dataset=rotoedit_lstm --test_batch_size=8 --max_tgt_length=1000 \
# 	--output_dir=/dataA/chenxiang/rotoedit_lstm_dual_copy \
# 	--beam_size=3 --tester=base-lstm_b3