---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen3-4B
tags:
- base_model:adapter:Qwen/Qwen3-4B
- lora
- transformers
pipeline_tag: summarization
model-index:
- name: qwen3-4B-finetuned-turkish-youtube-text-summarizer
  results: []
datasets:
- emirunlu26/turkish-youtube-text-summarization
language:
- tr
metrics:
- rouge
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3-4B-finetuned-turkish-youtube-text-summarizer

This model is a fine-tuned version of [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on [emirunlu26/turkish-youtube-text-summarization](https://huggingface.co/datasets/emirros/turkish-youtube-text-summarization) dataset.

## Limitations

- This model is developed as an individual project to learn the process of text preprocessing and fine-tuning a model with QLoRA. It is NOT for practical use.
- It is fine-tuned on 520 training samples only due to data scarcity issues.
- The model is fine-tuned on transcription and summary of videos which does not exceed 25 minutes.
  
## Model description

- Base Model: Qwen3-4B
- Fine-tuning Task: Turkish Text Summarization
- Fine-tuning Technique: QLoRA
- Fine-tuning Dataset: emirunlu26/turkish-youtube-text-summarization

## Performance

- ROUGE-1 (F1 score): 0.15
- ROUGE-2 (F1 score): 0.11
- ROUGE-L (F1 score): 0.11

## Usage

```python
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "Qwen/Qwen3-4B"
adapter_model_name = "emirunlu26/qwen3-4B-finetuned-turkish-youtube-text-summarizer"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name,device_map="cuda",torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, adapter_model_name).to("cuda")
model.eval()

def generate_prompt(sample):
  instruction_prompt = "Bu Youtube videosunu, ana teması ve önemli noktalarına odaklanarak kısa ama öz ve soyutlayıcı bir şekilde özetle (abstractive summary):\n"

  title = sample["title"]
  category = sample["category"]
  channel = sample["channel"]
  text = sample["text"]

  data_prompt = f"Başlık: {title}\n" \
  + f"Kategori: {category}\n" \
  + f"Kanal: {channel}\n" \
  + f"Metin: {text}"
  return (instruction_prompt + data_prompt)

def preprocess_sample(sample):
  prompt = generate_prompt(sample)
  messages = [
      {"role": "user", "content": prompt}
  ]

  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False
  )
  model_input = tokenizer([text],return_tensors="pt").to(model.device)
  return model_input

def generate_summary(model,model_input):
  generated_ids = model.generate(
      **model_input,
      max_new_tokens=2000
      )
  output_ids = generated_ids[0][len(model_input.input_ids[0]):].tolist()
  summary = tokenizer.decode(output_ids,skip_special_tokens=True).strip("\n")
  return summary

video_sample = {"title":<title>, "category":<category_name>, "channel":<channel_name>}

model_input = preprocess_sample(video_sample)
summary = generate_summary(model,model_input)



```

## QLoRA configurations

- rank = 32
- lora_alpha = 32
- lora_dropout = 0.05

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 2
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 15.3721       | 0.1923 | 50   | 12.8846         |
| 10.2132       | 0.3846 | 100  | 8.7292          |
| 8.1384        | 0.5769 | 150  | 7.7720          |
| 7.5919        | 0.7692 | 200  | 7.5426          |
| 7.4479        | 0.9615 | 250  | 7.4566          |
| 7.4262        | 1.1538 | 300  | 7.4169          |
| 7.397         | 1.3462 | 350  | 7.3958          |
| 7.3622        | 1.5385 | 400  | 7.3824          |
| 7.3669        | 1.7308 | 450  | 7.3723          |
| 7.3221        | 1.9231 | 500  | 7.3672          |


### Framework versions

- PEFT 0.16.0
- Transformers 4.53.2
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.21.2