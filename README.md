A BERT-Based Machine Reading Comprehension Baseline
------------------------------------------------------

This repository maintains a machine reading comprehension (MRC) baseline based on [BERT](https://arxiv.org/abs/1810.04805). We follow the system descriptions in the following two papers to implement this baseline. 

* [Improving Question Answering with External Knowledge](https://arxiv.org/abs/1902.00993)
* [Probing Prior Knowledge Needed in Challenging Chinese Machine Reading Comprehension](https://arxiv.org/abs/1904.09679)

If you find this code useful, please cite the following papers.

```
@article{pan2019improving,
  title={Improving Question Answering with External Knowledge},
  author={Pan, Xiaoman and Sun, Kai and Yu, Dian and Ji, Heng and Yu, Dong},
  journal={arXiv preprint arXiv:1902.00993},
  year={2019}
}

@article{sun2019probing,
  title={Probing Prior Knowledge Needed in Challenging Chinese Machine Reading Comprehension},
  author={Sun, Kai and Yu, Dian and Yu, Dong and Cardie, Claire},
  journal={arXiv preprint arXiv:1904.09679},
  year={2019}
}
```

Here, we show the usage of this baseline using a demo designed for the [DREAM](https://dataset.org/dream/) dataset.

  1. Download and unzip the pre-trained language model from https://github.com/google-research/bert. and set up the environment variable for BERT by ```export BERT_BASE_DIR=/PATH/TO/BERT/DIR```.
  2. Copy the data folder ```data``` from the [DREAM repo](https://github.com/nlpdata/dream) to ```bert/```.
  3. In ```bert```, execute ```python convert_tf_checkpoint_to_pytorch.py   --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin```
  4. Execute ```python run_classifier.py   --task_name dream  --do_train --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 2e-5   --num_train_epochs 8.0   --output_dir dream_finetuned  --gradient_accumulation_steps 3```
  5. The resulting fine-tuned model, predictions, and evaluation results are stored in ```bert/dream_finetuned```.

**Environment**: The code has been tested with Python 3.6 and PyTorch 1.0
