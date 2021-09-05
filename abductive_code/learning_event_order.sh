#CUDA_VISIBLE_DEVICES="1,2,3,4,5"
cd /users4/ldu/abductive/
python3 learning_event_order.py \
  --bert_model "/data/huggingface_transformers/roberta-large/" \
  --bert_tokenizer "/data/huggingface_transformers/roberta-large/" \
  --do_lower_case \
  --seed 6776 \
  --l2_reg 0.01 \
  --do_test \
  --train_data_dir "/users4/ldu/abductive/data/sentences_order_train_602000.pkl" \
  --test_data_dir "/users4/ldu/abductive/data/sentences_order_dev_600000_roc_sis.pkl" \
  --eval_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 40 \
  --output_dir "/users4/ldu/abductive/pretrained_models/event_order_pretrained/" \
  --num_frozen_epochs 0 \
  --start_layer 7 \
  --merge_layer 10 \
  --warmup_proportion 0.1 \
  --n_layer_extractor 2 \
  --n_layer_aa 3 \
  --n_layer_gnn 2 \
  --n_layer_merger 2 \
  --method_merger gat \
  --loss_aa_smooth 0.1 \
  --loss_aa_smooth_method diagnoal \
  --gpuls 4567 \
  --do_margin_loss \
  --margin 0.04 \
  --Lambda 0.01 \
  --pret \
  #--Lambda 0.01 \
  
  #--test_data_dir "/users4/ldu/GraphBert/data/mcnc/dat0/test_chains.pkl" \ !!!!!
  #--use_bert \
  #--layer_norm \
  #--sep_sent \
  #--use_bert \
  #--train_data_dir "/users4/ldu/GraphBert/data/mcnc/dat0/train_matched_chains.pkl" \
  #--pret

  
