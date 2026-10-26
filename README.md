# LLM-Finetuning-Hub

A collection of my LLM and SLM fine-tuning experiments, from parameter-efficient LoRA to full supervised fine-tuning (SFT) for structured extraction. Each experiment lives in its own folder with a notebook you can rerun. Trained models are published on my [Hugging Face profile](https://huggingface.co/ferozk0333).

## Experiments

| Experiment | Method | Base model | Task | Model |
|---|---|---|---|---|
| [`sft-finetuning`](./sft-finetuning) | Full SFT (TRL) | Gemma 3 270M-it | Extract food and drink items from text into a structured format | [Hugging Face](https://huggingface.co/ferozk0333/Text-Extraction-NER-FoodExtract-Gemma-3-270M-Fine-Tuned-v1) |
| [`lora-finetuning`](./lora-finetuning) | LoRA (PEFT), 8-bit Quantization | Gemma 7B | Customer support reply generation | Notebook |

## Featured: Food & Drink Extraction with Gemma 3 270M

A 270M-parameter model fine-tuned to turn raw text into a structured extraction.

**Input**

```
Hello, my name is Feroz!! I visited a beautiful eating place yesterday. It was a rustic wooden table set with a bowl of creamy tomato soup, a golden grilled cheese sandwich, and a tall glass of iced lemonade. However, when I got home, I also had a Diet Coke
```

**Output**

```
food_or_drink: 1
tags: fi
foods: creamy tomato soup, grilled cheese sandwich, iced lemonade
drinks: Diet Coke
```

**Gradio Demo Screenshot**
<img width="1893" height="723" alt="Screenshot 2026-10-25 at 8 24 11 PM" src="https://github.com/user-attachments/assets/5f3ec249-82ad-4ce1-bee5-b345b3f8b952" />


- **Dataset:** [mrdbourke/FoodExtract-1k](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k) (1,420 samples, 80/20 train/test split)
- **Training:** TRL `SFTTrainer`, prompt/completion format so loss is computed on the answer only
- **Result:** the base model ignores the format, while the fine-tuned model returns the structured output consistently

## Quick start

```python
from transformers import pipeline

gen = pipeline(
    "text-generation",
    model="ferozk0333/Text-Extraction-NER-FoodExtract-Gemma-3-270M-Fine-Tuned-v1",
    device_map="auto",
)
msg = [{"role": "user", "content": "A bowl of tomato soup with a glass of iced lemonade"}]
print(gen(msg, max_new_tokens=64, do_sample=False)[0]["generated_text"][-1]["content"])
```

Pass the raw text only, with no instruction prompt, since that is how the model was trained.

## Tech stack

PyTorch, Hugging Face Transformers, TRL, PEFT, Datasets, Gradio (demo)

## Repo structure

```
sft-finetuning/    # Gemma 3 270M, food extraction (SFT)
lora-finetuning/   # Gemma 7B, customer support (LoRA)
assets/            # demo screenshot
```

## Notes

Experiments are for learning and prototyping. Gemma models are subject to Google's [Gemma terms of use](https://ai.google.dev/gemma/terms).
