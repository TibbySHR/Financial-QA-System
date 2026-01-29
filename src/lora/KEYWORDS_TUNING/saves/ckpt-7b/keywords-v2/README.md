---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /udata/dingmin/LLaMA-Factory/models/Qwen1.5-7B/
model-index:
- name: keywords-v2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# keywords-v2

This model is a fine-tuned version of [/udata/dingmin/LLaMA-Factory/models/Qwen1.5-7B/](https://huggingface.co//udata/dingmin/LLaMA-Factory/models/Qwen1.5-7B/) on the keywords dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.11.1
- Transformers 4.41.2
- Pytorch 2.3.1+cu121
- Datasets 2.19.2
- Tokenizers 0.19.1