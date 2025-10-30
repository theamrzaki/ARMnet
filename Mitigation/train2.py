#--------------------------------------------------
#------------batch preprocessing-------------------
#--------------------------------------------------
import argparse
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

CRITERIA = ["effectiveness", "feasibility", "clarity", "completeness",
            "safety", "scalability", "efficiency", "compliance"]

# --- Keyword groups per criterion ---
KEYWORDS = {
    "feasibility": ["deploy", "implement", "configure", "monitor", "rollback", "execute", "apply"],
    "efficiency": ["reduce", "eliminate", "optimize", "resolve", "fix", "address", "root cause"],
    "scalability": ["automate", "distributed", "cluster", "load balancing", "replicate", "scale", "container"],
    "clarity": ["step", "first", "then", "finally", "procedure", "guide", "instruction"],
    "effectiveness": ["resolve", "address", "mitigate", "prevent", "eliminate", "fix", "improve"],
    "safety_penalty": ["crash", "failure", "error", "downtime", "breach"],
    "compliance": ["policy", "standard", "guideline", "requirement", "audit"],
}


CONFIG = {
    "root_cause_file": "data/root_causes.json",
    "max_causes": 100,
    "num_train_epochs": 6,
    "logging_steps": 10,
    "max_new_tokens": 64,
    "temperature": 0.9,
}

def reward_function(completions, flags=None, prompts=None, normalize=True, **kwargs) -> list:
    """
    Custom reward function with per-component flags.
    Each flag enables/disables a scoring component.
    
    Args:
        completions (list[str]): generated texts
        flags (dict): e.g., {'feasibility': True, 'clarity': False}
        prompts (list[str], optional)
        **kwargs: extra unused args for compatibility
    """
    # default: enable all criteria

    flags = flags or {k: True for k in CRITERIA}

    rewards = []
    for c in completions:
        text = c.lower()

        # ---- 1️⃣ Effectiveness ----
        eff = min(sum(text.count(k) for k in KEYWORDS["effectiveness"]), 10) if flags.get("effectiveness", True) else 0

        # ---- 2️⃣ Feasibility ----
        feas = min(sum(text.count(k) for k in KEYWORDS["feasibility"]), 10) if flags.get("feasibility", True) else 0

        # ---- 3️⃣ Clarity ----
        if flags.get("clarity", True):
            clarity = max(0, min(5, flesch_reading_ease(c) / 20))
        else:
            clarity = 0

        # ---- 4️⃣ Completeness (unique actions) ----
        if flags.get("completeness", True):
            unique_actions = set(re.findall(r'\b(optimize|reduce|mitigate|prevent|fix|monitor|update|check)\b', text))
            complete = min(len(unique_actions), 5)
        else:
            complete = 0

        # ---- 5️⃣ Safety ----
        safety = (5 - min(sum(text.count(k) for k in KEYWORDS["safety_penalty"]), 5)) if flags.get("safety", True) else 0

        # ---- 6️⃣ Scalability ----
        scale = min(sum(text.count(k) for k in KEYWORDS["scalability"]), 5) if flags.get("scalability", True) else 0

        # ---- 7️⃣ Efficiency ----
        efficiency = min(sum(text.count(k) for k in KEYWORDS["efficiency"]), 5) if flags.get("efficiency", True) else 0

        # ---- 8️⃣ Compliance ----
        compliance = min(sum(text.count(k) for k in KEYWORDS["compliance"]), 5) if flags.get("compliance", True) else 0

        # ---- Final weighted sum ----
        reward = (
            0.25 * eff +
            0.20 * feas +
            0.10 * clarity +
            0.10 * complete +
            0.10 * safety +
            0.05 * scale +
            0.10 * efficiency +
            0.10 * compliance
        )

        rewards.append(reward)

    return rewards

def make_reward_for_experiment(experiment="full", leave_out_crit=None, only_one_crit=None, active_criteria=None, normalize=False):
    """
    Returns a reward function for GRPOTrainer based on the experiment.
    
    Args:
        experiment: "full" or "leave_one_out"
        leave_out_crit: (str) the criterion to leave out if experiment=="leave_one_out"
        normalize: bool, normalize individual components
    """
    # All criteria flags
    all_flags = {k: True for k in CRITERIA}    

    if experiment == "leave_one_out":
        if leave_out_crit is None or leave_out_crit not in CRITERIA:
            raise ValueError(f"leave_out_crit must be one of {CRITERIA}")
        # disable the chosen criterion
        all_flags[leave_out_crit] = False

    elif experiment == "only_one":
        if only_one_crit is None or only_one_crit not in CRITERIA:
            raise ValueError(f"only_one_crit must be one of {list(CRITERIA)}")
        # disable all except the selected one
        all_flags = {k: (k == only_one_crit) for k in CRITERIA}

    elif experiment == "custom":
        active_criteria = active_criteria.split(",") if isinstance(active_criteria, str) else active_criteria
        all_flags = {k: (k in active_criteria) for k in CRITERIA}

    def reward_fn(completions, prompts=None, **kwargs):
        return reward_function(
            completions,
            flags=all_flags,
            **kwargs
        )
    
    return reward_fn

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

def run(arguments=None):
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
    
    if arguments is not None:
        experiment = arguments.experiment
        leave_out_crit = arguments.leave_out_crit
        normalize = arguments.normalize

    # Create save directory
    os.makedirs("./saved_models", exist_ok=True)

    if experiment == "full" :
        reward_full = make_reward_for_experiment(experiment="full")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_full],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained("./saved_models/full_reward")
        tokenizer.save_pretrained("./saved_models/full_reward")
    elif experiment == "leave_one_out":
        reward_loo = make_reward_for_experiment(experiment="leave_one_out", leave_out_crit=arguments.leave_out_crit)
    
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_loo],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )

        save_dir = f"./saved_models/leave_out_{arguments.leave_out_crit}"
        os.makedirs(save_dir, exist_ok=True)
        
        trainer.train()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    elif experiment == "only_one":
        reward_one = make_reward_for_experiment(
            experiment="only_one",
            only_one_crit=arguments.only_one_crit,
        )

        save_dir = f"./saved_models/only_one_{arguments.only_one_crit}"
        os.makedirs(save_dir, exist_ok=True)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_one],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    elif experiment == "custom":
        reward_custom = make_reward_for_experiment(
            experiment="custom",
            active_criteria=arguments.active_criteria,
        )

        save_dir = f"./saved_models/custom_{'_'.join(arguments.active_criteria)}"
        os.makedirs(save_dir, exist_ok=True)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_custom],
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="full")
    parser.add_argument("--leave_out_crit", type=str, default=None)
    parser.add_argument("--only_one_crit", type=str, default=None)
    parser.add_argument("--active_criteria", type=str, nargs='*', default=None)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    run(arguments=args)