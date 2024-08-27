import os
import gc
import torch
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import (
    LoraConfig, 
    PeftModel, 
)
from trl import DPOTrainer
import bitsandbytes as bnb
from utils.metrics import compute_metrics_fn

# Avoid storage leaks
torch.cuda.empty_cache()

def fine_tune(base_model_name: str, train_dataset: Dataset, test_dataset: Dataset, new_model: str, device_map: dict[str, int]):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Check GPU compatibility with bfloat16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
    elif torch.backends.mps.is_available():
        print("=" * 80)
        print("Running on Apple Silicon GPU with MPS backend.")
        print("=" * 80)
    else:
        print("No GPU backend available. Running on CPU.")

    # Load base model
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=None,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=None,
            device_map=device_map
        )
        # Reference model
        ref_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map=device_map,
        )

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    chat_template = open('llama-3-instruct.jinja').read()
    chat_template = chat_template.replace('    ', '').replace('\n', '')

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_steps=50, #tweak this to change # of steps in the training run
        save_strategy="no",
        logging_steps=1,
        output_dir=new_model,
        optim="paged_adamw_32bit",
        warmup_steps=10,
        bf16=True,
        # report_to="wandb",
    )

    # Set supervised fine-tuning parameters
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,
        max_prompt_length=None,
        max_length=1024, 
        force_use_ref_model=True
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(new_model + "-adapter")
    tokenizer.save_pretrained(new_model + "-adapter")

    # Empty VRAM
    del trainer, model, ref_model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.chat_template = chat_template

    # Merge models
    model = PeftModel.from_pretrained(base_model, new_model + "-adapter")
    model = model.merge_and_unload()

    # Save model and tokenizer
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)
