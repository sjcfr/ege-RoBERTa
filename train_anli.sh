


cd /users4/ldu/abductive/
nohup python3 Train_anli.py \
  --bert_model "/users4/ldu/abductive/pretrained_models/graph_pretrained/e_11902vgb_c_b_I_a_7_10_2_3_0_2_0_a_0.01_F_F_c_g_2e-05_0.1_0.0_um687774.pkl" \
  --bert_tokenizer "/data/huggingface_transformers/roberta-base/" \
  --do_lower_case \
  --seed 6776 \
  --l2_reg 0.01 \
  --do_test \
  --train_data_dir "/users4/ldu/abductive/data/train_pret.pkl" \
  --test_data_dir "/users4/ldu/abductive/data/dev_pret.pkl" \
  --eval_batch_size 24 \
  --learning_rate 2e-5 \
  --num_train_epochs 6 \
  --max_seq_length 50 \
  --output_dir "/users4/ldu/abductive/models/" \
  --num_frozen_epochs 0 \
  --start_layer 7 \
  --merge_layer 10 \
  --warmup_proportion 0.1 \
  --n_layer_extractor 2 \
  --n_layer_aa 3 \
  --n_layer_gnn 0 \
  --n_layer_merger 2 \
  --method_merger gat \
  --method_gnn skip \
  --loss_aa_smooth 0.1 \
  --loss_aa_smooth_method diagnoal \
  --gpuls 1340 \
  --margin 0.04 \
  --Lambda 0.01 \
  --pret \
  --model_type 'vgb_c' &
  
#"
