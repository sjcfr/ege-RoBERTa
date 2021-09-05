# ege-RoBERTa
Learning event graph knowledge for abductive reasoning.

To this end, we involve a two stage learning process to introduce the event graph knowledge, and a variational autoencoder based model ege-RoBERTa to capture the event graph knowledge. 

### Pre-training Stage: 
**Learning Event Graph Knowledge from a Pseudo Instance Set**

* Preprocess datasets using pret.py
* Using event_order_gen.py to sample adjacent and non-adjacent event pairs for traing a next event prediction model. Then run learning_event_order.sh to train the next event prediction model (described in the Sec 5.2 of original paper).
* Using aux_dat_gen.py to construct the pseudo instance set. 
* Then pretrain.sh is used for conducting the first stage training process. 

### Finetuning Stage: 
** Adapt Event Graph Knowledge to the Abductive Reasoning Task**

Please refer to train_anli.sh and Train_anli.py.

### Model Architecture

Files for constructing model architecture is contained in the file folder onmt.


