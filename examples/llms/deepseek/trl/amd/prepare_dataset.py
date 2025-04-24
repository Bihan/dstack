from datasets import Dataset, load_dataset


def make_prompt_gpt_response_dataset(
    dataset_name: str,
    split: str = "train",
    output_path: str | None = None,
) -> Dataset:
    """
    Load a trajectory-style dataset and break it into prompt/gpt_response pairs.

    Each GPT turn becomes one example:
      - `prompt`: all messages (human+gpt) before that turn, joined with newlines.
      - `gpt_response`: the GPT message value at that turn.
      - `loss`: the GPT turn's `loss` flag.
      - any other metadata (e.g. `item_id`).

    Optionally saves the new dataset to disk.
    """
    # 1. Load the raw trajectory dataset
    # raw = load_dataset(dataset_name, split=split)
    raw_full = load_dataset(dataset_name, split="train")
    raw = raw_full.filter(lambda ex: ex.get("item_id", "").startswith("webshop"))

    samples = []
    for ex in raw:
        # carry through item_id or other metadata fields
        metadata = {k: ex[k] for k in ex if k not in ("conversations",)}
        conv = ex["conversations"]
        # iterate through each turn
        for idx, msg in enumerate(conv):
            if msg.get("from") == "gpt":
                # build prompt from all previous messages
                prompt = "\n".join(f"{m['from']}: {m['value']}" for m in conv[:idx])
                gpt_response = msg.get("value", "")
                loss_flag = msg.get("loss")
                sample = {
                    "prompt": prompt,
                    "gpt_response": gpt_response,
                    "loss": loss_flag,
                    **metadata,
                }
                samples.append(sample)

    # 2. Create a new HuggingFace dataset
    new_ds = Dataset.from_list(samples)

    # For testing
    # for idx, example in enumerate(new_ds):
    #     if idx >= 5:
    #         break
    #     print(f"\n--- Example {idx} ---")
    #     print("prompt:")
    #     print(example.get("prompt"))
    #     if "gpt_response" in example:
    #         print("gpt_response:")
    #         print(example.get("gpt_response"))
    #     else:
    #         # no separate gpt_response column, print full record
    #         print("full record:", example)

    # 3. Optionally save
    if output_path:
        new_ds.save_to_disk(output_path)
    return new_ds


if __name__ == "__main__":
    # Example usage
    ds = make_prompt_gpt_response_dataset(
        dataset_name="AgentGym/AgentTraj-L",
        split="train",
        output_path="./AgentTraj_L_prompt_gpt_response",
    )
    print(f"Created dataset with {len(ds)} examples")
