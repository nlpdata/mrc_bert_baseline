A demo of our machine reading comprehension baseline based on BERT
------------------------------------------------------------------

This repository maintains an MRC baseline based on BERT, whose implemetation is based on the baseline description in the following two works

* [Improving Question Answering with External Knowledge](https://arxiv.org/abs/1902.00993)
* [Probing Prior Knowledge Needed in Challenging Chinese Machine Reading Comprehension](https://arxiv.org/abs/1904.09679)



Here, we show the usage of our baseline by a demo designed for the [DREAM](https://dataset.org/dream/) dataset.

  1. Download and unzip the pre-trained language model from https://github.com/google-research/bert. and set up the environment variable for BERT by ```export BERT_BASE_DIR=/PATH/TO/BERT/DIR```.
  2. Copy the data folder ```data``` from the [DREAM repo](https://github.com/nlpdata/dream) to ```ftlm++/```.
  3. In ```bert```, execute ```python convert_tf_checkpoint_to_pytorch.py   --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin```
  4. Execute ```python run_classifier.py   --task_name dream  --do_train --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir dream_finetuned  --gradient_accumulation_steps 3```
  5. The resulting fine-tuned model, predictions, and evaluation results are stored in ```bert/dream_finetuned```.

**Environment**: The code has been tested with Python 3.6 and PyTorch 1.0
