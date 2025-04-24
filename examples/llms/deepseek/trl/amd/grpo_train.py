import argparse
import re
from typing import List

from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using GRPOTrainer.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device for training",
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
        "--num_generations",
        type=int,
        default=2,
        help="Number of generations per prompt (must divide global batch size)",
    )
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps interval")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory for the trained model"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Trust remote code when loading the model"
    )
    return parser.parse_args()


def reward_len(completions, **kwargs):
    return [abs(20 - len(completion)) for completion in completions]


def reward_fn(
    prompts: List[str], completions: List[str], gpt_response: List[str], loss: List[bool], **kwargs
) -> List[float]:
    """
    Custom reward function with half‐credit for bad examples that still attempt an action.

    Args:
      prompts:      list of input contexts (unused).
      completions:  list of generated model strings.
      gpt_response: list of ground‐truth GPT turns from the dataset.
      loss:         list of booleans (True if example was marked “bad”).

    Returns:
      A list of rewards, one per example:
        * If loss == True:
            - 0.5 if the model’s completion contains ANY click[...] or search[...] pattern
            - otherwise 0.0
        * Else (loss == False):
            - +2 / −2 for exact click[...] matches vs. mismatches
            - +1 / −1 for any search[...] matches vs. mismatches
            - 0.0 if no recognizable action in the ground truth
    """
    rewards: List[float] = []
    click_pat = re.compile(r"click\[[^\]]+\]")
    search_pat = re.compile(r"search\[[^\]]*\]")

    for gen, loss_flag, gt in zip(completions, loss, gpt_response):
        gen = gen or ""
        gt = gt or ""

        # 1) Bad example? half‐credit if at least an action was attempted.
        if loss_flag:
            if click_pat.search(gen) or search_pat.search(gen):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
            continue

        # 2) Good example: check ground‐truth click[...] first
        m_click = click_pat.search(gt)
        if m_click:
            want = m_click.group(0)  # e.g. "click[B09PYSKD7H]"
            m_gen = click_pat.search(gen)
            if m_gen and m_gen.group(0) == want:
                rewards.append(2.0)
            else:
                rewards.append(-2.0)
            continue

        # 3) Good example: check ground‐truth search[...]
        if search_pat.search(gt):
            if search_pat.search(gen):
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
            continue

        # 4) No action expected → neutral
        rewards.append(0.0)

    return rewards


def main():
    args = parse_args()

    # dataset = load_dataset(args.dataset_name, split="train")
    dataset = Dataset.from_file(
        "/workflow/grpo_example/AgentTraj_L_prompt_gpt_response/data-00000-of-00001.arrow"
    )
    print(dataset)

    # Keep only examples whose item_id starts with 'webshop'
    # dataset = dataset.filter(lambda ex: ex.get("item_id", "").startswith("webshop"), batched=False)

    # # Convert conversations into a textual prompt
    # def make_prompt(example):
    #     lines = [f"{m['from']}: {m['value']}" for m in example['conversations']]
    #     return { 'prompt': "\n".join(lines) }
    # dataset = dataset.map(make_prompt, batched=False)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    trainer = GRPOTrainer(
        model=model, reward_funcs=reward_fn, args=training_args, train_dataset=dataset
    )

    trainer.train()


if __name__ == "__main__":
    main()
