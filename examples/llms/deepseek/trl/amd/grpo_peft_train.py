import argparse
import re
from typing import List

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model using GRPOTrainer with PEFT LoRA and custom trajectory reward."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path or HF ID of the pretrained model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device for training",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=2,
        help="Number of generations per prompt (must divide global batch size)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging steps interval",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save a checkpoint every N steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for the trained model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading the model",
    )
    # LoRA hyperparameters
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target modules for LoRA adaptation",
    )
    return parser.parse_args()


# data = {
#             "prompt": prompt,
#             "completion": completion,
#             "loss": l,
#             "gpt_resp": gpt_response
#         }

#         print(json.dumps(data, indent=2))


def reward_fn(
    prompts: List[str], completions: List[str], gpt_response: List[str], loss: List[bool], **kwargs
) -> List[float]:
    """
    Custom reward function:

      * If loss == True:
          - 0.5 if the model’s completion contains ANY click[...] or search[...] pattern
          - otherwise 0.0

      * Else (loss == False):
          - +2.0 if ground‐truth has click[...] and the model’s completion has the identical click[...]
          - +1.0 if ground‐truth has search[...] and the model’s completion has any search[...]
          - +0.25 if neither pattern expected but the completion contains both the words "search" and "click"
          - otherwise 0.0
    """
    click_pat = re.compile(r"click\[[^\]]+\]")
    search_pat = re.compile(r"search\[[^\]]*\]")

    rewards: List[float] = []
    for completion, loss_flag, gt in zip(completions, loss, gpt_response):
        completion = completion or ""
        gt = gt or ""

        # 1) Bad example? half‐credit if any attempt at an action
        if loss_flag:
            # if click_pat.search(completion) or search_pat.search(completion):
            #     rewards.append(0.5)
            # else:
            rewards.append(0.0)
            continue

        # 2) Good example: look for exact click[...] in the ground truth
        m_click = click_pat.search(gt)
        if m_click:
            want = m_click.group(0)  # e.g. "click[B09PYSKD7H]"
            m_comp = click_pat.search(completion)
            rewards.append(2.0 if (m_comp and m_comp.group(0) == want) else 0.0)
            continue

        # 3) Good example: look for any search[...] in the ground truth
        if search_pat.search(gt):
            rewards.append(1.0 if search_pat.search(completion) else 0.0)
            continue

        # 4) No strict pattern expected but both keywords appear
        if "search" in completion and "search" in gt:
            rewards.append(0.25)
            continue

        if "click" in completion and "click" in gt:
            rewards.append(0.25)
            continue

        if "search" in completion or "click" in completion:
            rewards.append(0.1)
            continue

        # 5) Otherwise, neutral
        rewards.append(0.0)
    print(rewards)
    return rewards


def reward_fn_v1(
    prompts: List[str], completions: List[str], gpt_response: List[str], loss: List[bool], **kwargs
) -> List[float]:
    """
    Custom reward function:

      * If loss == True:
          - 0.5 if the model’s completion contains ANY click[...] or search[...] pattern
          - otherwise 0.0

      * Else (loss == False):
          - +2.0 if ground‐truth has click[...] and the model’s completion has the identical click[...]
          - +1.0 if ground‐truth has search[...] and the model’s completion has any search[...]
          - +0.25 if neither pattern expected but the completion contains both the words "search" and "click"
          - otherwise 0.0
    """
    click_pat = re.compile(r"click\[[^\]]+\]")
    search_pat = re.compile(r"search\[[^\]]*\]")

    rewards: List[float] = []
    for completion, loss_flag, gt in zip(completions, loss, gpt_response):
        completion = completion or ""
        gt = gt or ""

        # 1) Bad example? half‐credit if any attempt at an action
        if loss_flag:
            if click_pat.search(completion) or search_pat.search(completion):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
            continue

        # 2) Good example: look for exact click[...] in the ground truth
        m_click = click_pat.search(gt)
        if m_click:
            want = m_click.group(0)  # e.g. "click[B09PYSKD7H]"
            m_comp = click_pat.search(completion)
            rewards.append(2.0 if (m_comp and m_comp.group(0) == want) else 0.0)
            continue

        # 3) Good example: look for any search[...] in the ground truth
        if search_pat.search(gt):
            rewards.append(1.0 if search_pat.search(completion) else 0.0)
            continue

        # 4) No strict pattern expected but both keywords appear
        if "search" in completion and "click" in completion:
            rewards.append(0.25)
            continue

        # 5) Otherwise, neutral
        rewards.append(0.0)
    print(rewards)
    return rewards


def main():
    args = parse_args()

    # Load dataset (must contain 'conversations' column)
    # dataset = load_dataset(args.dataset_name, split="train")
    dataset = Dataset.from_file("./AgentTraj_L_prompt_gpt_response/data-00000-of-00001.arrow")

    # GRPO training config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        report_to="wandb",
    )

    # 8-bit quantization
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model in 8-bit
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
