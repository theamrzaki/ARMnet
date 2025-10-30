#--------------------------------------------------
#------------batch preprocessing-------------------
#--------------------------------------------------
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from datasets import Dataset
from sklearn.model_selection import train_test_split
from textstat import flesch_reading_ease
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, GRPOTrainer
import torch
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging
import os
from datetime import datetime
import json
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig
from peft import prepare_model_for_kbit_training

# ==================================================
# 1. Reward Function
# ==================================================
KEYWORDS = {
    "effectiveness": ["mitigate", "reduce", "prevent", "optimize"],
    "feasibility": ["simple", "easy", "implementable", "low-cost"],
    "safety_penalty": ["may fail", "could break", "risk", "danger"],
    "scalability": ["any system", "generally applicable", "scalable"],
    "efficiency": ["efficient", "optimize", "low resource", "cost-effective"],
    "compliance": ["policy", "standard", "guideline", "law"]
}

CONFIG = {
    "root_cause_file": "data/root_causes.json",
    "max_causes": 100,
    "num_train_epochs": 6,
    "logging_steps": 10,
    "max_new_tokens": 64,
    "temperature": 0.9,
}

def reward_function(completions, prompts=None, **kwargs) -> list:
    rewards = []
    for c in completions:
        text = c.lower()
        eff = min(sum(text.count(k) for k in KEYWORDS["effectiveness"]), 10)
        feas = min(sum(text.count(k) for k in KEYWORDS["feasibility"]), 10)
        clarity = max(0, min(5, flesch_reading_ease(c) / 20))
        unique_actions = set(re.findall(r'\b(optimize|reduce|mitigate|prevent|fix|monitor|update|check)\b', text))
        complete = min(len(unique_actions), 5)
        safety = 5 - min(sum(text.count(k) for k in KEYWORDS["safety_penalty"]), 5)
        scale = min(sum(text.count(k) for k in KEYWORDS["scalability"]), 5)
        efficiency = min(sum(text.count(k) for k in KEYWORDS["efficiency"]), 5)
        compliance = min(sum(text.count(k) for k in KEYWORDS["compliance"]), 5)

        reward = (
            0.25*eff + 0.2*feas + 0.1*clarity + 0.1*complete +
            0.1*safety + 0.05*scale + 0.1*efficiency + 0.1*compliance
        )
        rewards.append(reward)
    return rewards



#--------------------------------------------------
#------------------TensorBoard Training Loop--------
#--------------------------------------------------

def load_root_causes(path, limit=100):
    with open(path, "r") as f:
        data = json.load(f)
    
    causes = []
    for key, items in data.items():
        for cause in items[:limit]:
            causes.append({
                "prompt": f"Root cause: {cause}. Suggest a clear, feasible, and effective mitigation plan.",
                "solution": ""  # Leave empty if RLHF will fill it later
            })
    return causes

def generate_plan(generator, cause: str, max_new_tokens: int = 256, temperature: float = 0.7) -> dict:
    """
    Generate a mitigation plan for a given root cause using the provided generator.
    Returns both the prompt and completion.
    """
    prompt = f"Root cause: {cause}. Suggest a clear, feasible, and effective mitigation plan."
    
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
    )

    full_text = outputs[0]["generated_text"]
    completion = full_text.replace(prompt, "").strip()

    return {"prompt": prompt, "completion": completion}

def prepare_dataset(causes, generator, reward_function, max_new_tokens=256, temperature=0.7):
    """
    Prepare dataset for RLHF training by generating mitigation plans and assigning rewards.
    """
    dataset = []

    for cause in tqdm(causes, desc="Generating mitigation plans"):
        result = generate_plan(generator, cause, max_new_tokens, temperature)
        prompt = result["prompt"]
        completion = result["completion"]

        # Compute the reward based on the generated mitigation plan
        reward = reward_function(prompt, completion)

        dataset.append({
            "prompt": prompt,
            "solution": completion,  # "solution" is what GRPOTrainer expects
            "reward": reward,
        })

        # Optional cleanup to avoid GPU memory leaks
        gc.collect()
        torch.cuda.empty_cache()

    return dataset

#-----------------------------------------------
#-------------------Train PPO-------------------
#-----------------------------------------------

from peft import prepare_model_for_kbit_training


from peft import prepare_model_for_kbit_training

def run():
    # Load root causes
    dataset = load_root_causes("/home/db2003/Desktop/Amr/Tests/RCAEval/adservice_cpu_1_baro_re1_ob.json", limit=100)  
    train_data, test_data = train_test_split(
        dataset,
        test_size=0.01,
        random_state=42
    )
    train_data = Dataset.from_list(train_data)
    test_data = Dataset.from_list(test_data)
    
    # Use 4-bit quantization with FP16 compute dtype
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Changed from bfloat16 to float16
    )
    
    # Initialize TensorBoard
    log_dir = f"logs/ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Load model and tokenizer
    model_id = "ibm-granite/granite-4.0-micro"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,  # Changed from bfloat16 to float16
        attn_implementation="eager",
        device_map="auto",
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    
    # Enable input gradients
    model.enable_input_require_grads()
    
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Configure training arguments using GRPOConfig
    training_args = GRPOConfig(
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        gradient_accumulation_steps=4,
        num_train_epochs=12,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,   # Changed to True
        bf16=False,  # Changed to False
        max_completion_length=128,
        num_generations=8,
        max_prompt_length=128,
        max_grad_norm=1.0,
        temperature=0.7,
        top_p=0.9,
        report_to=["tensorboard"],
        logging_steps=10,
        push_to_hub=False,
        save_strategy="steps",
        dataloader_drop_last=True,
        optim="paged_adamw_8bit",
    )
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
    )
    
    # Create save directory
    os.makedirs("./saved_models", exist_ok=True)
    trainer.train()
    model.save_pretrained("./saved_models/final_model_3")
    tokenizer.save_pretrained("./saved_models/final_model_3")

if __name__ == "__main__":

    run()