# Deepseek
This example walks you through how to deploy and train Deepseek models with `dstack`. Here we have used Deepseek-R1 distilled models as well as Deepseek-V2-Lite, a 16B model that resembles the architecture of Deepseek-R1 (671B). 

Deepseek-V2-Lite uses the same MLA(Multi-head Latent Attention) and DeepSeekMoE (Mixture-of-Experts) as Deepseek-R1, but requires less memory which is ideal for testing inference and fine-tuning on smaller GPUs.

??? info "Prerequisites"
    Once `dstack` is [installed](https://dstack.ai/docs/installation), go ahead clone the repo, and run `dstack init`.

    <div class="termy">
 
    ```shell
    $ git clone https://github.com/dstackai/dstack
    $ cd dstack
    $ dstack init
    ```
    </div>

## Deployment
### AMD
Here's an example of a service that deploys Deepseek-R1-Distill-Llama-70B using `SGLang` and `vLLM` with AMD `Mi300x` GPU. The below configurations also support Deepseek-V2-Lite.

=== "SGLang"

    <div editor-title="examples/llms/deepseek/sglang/amd/.dstack.yml">
    ```yaml
    type: service
    name: deepseek-r1-amd

    image: lmsysorg/sglang:v0.4.1.post4-rocm620
    env:
      - MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-70B
    commands:
       - python3 -m sglang.launch_server
         --model-path $MODEL_ID
         --port 8000
         --trust-remote-code

    port: 8000
    model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B

    resources:
        gpu: mi300x
        disk: 300Gb

    ```
    </div>

=== "vLLM"
    
    <div editor-title="examples/llms/deepseek/sglang/amd/.dstack.yml">
    ```yaml
    type: service
    name: deepseek-r1-amd
    
    image: rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
    env:
      - MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-70B
      - MAX_MODEL_LEN=126432
    commands:
      - vllm serve $MODEL_ID
        --max-model-len $MAX_MODEL_LEN
        --trust-remote-code
    port: 8000
    
    model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
    
    
    resources:
        gpu: mi300x
        disk: 300Gb
    ```
    </div>

Note, when using Deepseek-70B with vLLM with a 192GB GPU, we must limit the context size to 126432 tokens to fit the memory.

### INTEL
Here's an example of a service that deploys Deepseek-R1-Distill-Llama-70B using `TGI` and `vLLM` with Intel Gaudi2 GPUs. Both TGI and vLLM does not support Deepseek-V2-Lite.

=== "TGI"

    <div editor-title="examples/llms/deepseek/tgi/intel/.dstack.yml">
    ```yaml
    type: service

    name: tgi

    image: ghcr.io/huggingface/tgi-gaudi:2.3.1

    auth: false
    port: 8000

    model: DeepSeek-R1-Distill-Llama-70B

    env:
      - HF_TOKEN
      - MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-70B
      - PORT=8000
      - OMPI_MCA_btl_vader_single_copy_mechanism=none
      - TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true
      - PT_HPU_ENABLE_LAZY_COLLECTIVES=true
      - MAX_TOTAL_TOKENS=2048
      - BATCH_BUCKET_SIZE=256
      - PREFILL_BATCH_BUCKET_SIZE=4
      - PAD_SEQUENCE_TO_MULTIPLE_OF=64
      - ENABLE_HPU_GRAPH=true
      - LIMIT_HPU_GRAPH=true
      - USE_FLASH_ATTENTION=true
      - FLASH_ATTENTION_RECOMPUTE=true

    commands:
      - text-generation-launcher
          --sharded true
          --num-shard 8
          --max-input-length 1024
          --max-total-tokens 2048
          --max-batch-prefill-tokens 4096
          --max-batch-total-tokens 524288
          --max-waiting-tokens 7
          --waiting-served-ratio 1.2
          --max-concurrent-requests 512

    resources:
      gpu: Gaudi2:8
    ```
    </div>

=== "vLLM"

    <div editor-title="examples/llms/deepseek/vllm/intel/.dstack.yml">
    ```yaml
    type: service
    name: deepseek-r1-gaudi

    image: vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0
    env:
      - MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-70B
    commands:
      - git clone https://github.com/vllm-project/vllm.git
      - cd vllm
      - pip install -r requirements-hpu.txt
      - python setup.py develop
      - vllm serve $MODEL_ID
        --tensor-parallel-size 8

    port: 8000

    model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B

    resources:
        gpu: gaudi2:8
    ```
    </div>  

### NVIDIA
Here's an example of a service that deploys Deepseek-R1-Distill-Llama-8B using `SGLang` and `vLLM` with NVIDIA GPUs. To run Deepseek-V2-Lite with vLLM, we must use 40GB GPU and limit the context size to 4096 tokens. Similarly, to run Deepseek-V2-Lite with SGLang, we must use 80GB GPU as per this [issue](https://github.com/sgl-project/sglang/issues/3451).

=== "SGLang"

    <div editor-title="examples/llms/deepseek/sglang/nvidia/.dstack.yml">
    ```yaml
    type: service
    name: deepseek-r1-nvidia
    
    image: lmsysorg/sglang:latest
    env:
      - MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    commands:
        - python3 -m sglang.launch_server
          --model-path $MODEL_ID
          --port 8000
          --trust-remote-code
    
    port: 8000
    
    model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    
    resources:
        gpu: 24GB
    ```
    </div>

=== "vLLM"

    <div editor-title="examples/llms/deepseek/vllm/nvidia/.dstack.yml">
    ```yaml
    type: service
    name: deepseek-r1-nvidia
    
    image: vllm/vllm-openai:latest
    env:
      - MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
      - MAX_MODEL_LEN=4096
    commands:
      - vllm serve $MODEL_ID
        --max-model-len $MAX_MODEL_LEN
    
    port: 8000
    
    model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    
    resources:
        gpu: 24GB
    ```
    </div>

Note, when using Deepseek-8B with vLLM with a 24GB GPU, we must limit the context size to 4096 tokens to fit the memory.

### Memory requirements

Below are the approximate memory requirements for loading the model. 
This excludes memory for the model context and CUDA/ROCm kernel reservations.

| Model Name               | Model size | FP16    | FP8     | INT4    |
|--------------------------|------------|---------|---------|---------|
| Deepseek-R1              | **671B**   | ~1342GB | ~671GB  | ~336GB  |
| DeepSeek-R1-Distill-Llama| **70B**    | ~161GB  | ~80.5GB | ~40B    |
| DeepSeek-R1-Distill-Qwen | **32B**    | ~74GB   | ~37GB   | ~18.5GB |
| DeepSeek-V2-Lite         | **16B**    | ~35GB   | ~17.5GB | ~8.75GB |
| DeepSeek-R1-Distill-Qwen | **14B**    | ~32GB   | ~16GB   | ~8GB    |
| DeepSeek-R1-Distill-Llama| **8B**     | ~18GB   | ~9GB    | ~4.5GB  |
| DeepSeek-R1-Distill-Qwen | **7B**     | ~16GB   | ~8GB    | ~4GB    |
| DeepSeek-R1-Distill-Qwen | **1.5B**   | ~3.5GB  | ~2GB    | ~1GB    |
 
For example, the FP8 version of Deepseek-R1 671B fits into single node of `Mi300x` with eight 192GB GPUs or 
singe node of `H200` with eight 141GB GPUs.


### Running a configuration

To run a configuration, use the [`dstack apply`](https://dstack.ai/docs/reference/cli/dstack/apply.md) command.

<div class="termy">

```shell
$ dstack apply -f examples/llms/deepseek/sglang/amd/.dstack.yml

 #  BACKEND  REGION     RESOURCES                         SPOT  PRICE   
 1  runpod   EU-RO-1   24xCPU, 283GB, 1xMI300X (192GB)    no    $2.49  
    
Submit the run deepseek-r1-amd? [y/n]: y

Provisioning...
---> 100%
```
</div>

Once the service is up, the model will be available via the OpenAI-compatible endpoint
at `<dstack server URL>/proxy/models/<project name>/`.

<div class="termy">

```shell
curl http://127.0.0.1:3000/proxy/models/main/chat/completions \
    -X POST \
    -H 'Authorization: Bearer &lt;dstack token&gt;' \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "What is Deep Learning?"
        }
      ],
      "stream": true,
      "max_tokens": 512
    }'
```
</div>


When a [gateway](https://dstack.ai/docs/concepts/gateways.md) is configured, the OpenAI-compatible endpoint 
is available at `https://gateway.<gateway domain>/`.

## Fine-tuning
### AMD
Here are examples of LoRA and GRPO fine-tuning of DeepSeek-R1-Distill-Qwen-1.5B on `Mi300x` GPU.

=== "LoRA"

    <div editor-title="examples/llms/deepseek/trl/amd/.dstack.yml">
    ```yaml
    type: dev-environment
    # The name is optional, if not specified, generated randomly
    name: trl-train

    image: rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0

    ide: vscode
    # Required environment variables
    env:
      - WANDB_API_KEY
      - WANDB_PROJECT
      - MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

    # Commands of the task
    commands:
      - git clone https://github.com/huggingface/trl.git
      - pip install transformers
      - pip install trl
      - pip install peft
      - pip install wandb
      - cd trl/trl/scripts
      - python sft.py
        --model_name_or_path $MODEL_ID
        --dataset_name trl-lib/Capybara
        --learning_rate 2.0e-4
        --num_train_epochs 1
        --packing
        --per_device_train_batch_size 2
        --gradient_accumulation_steps 8
        --gradient_checkpointing
        --logging_steps 25
        --eval_strategy steps
        --eval_steps 100
        --use_peft
        --lora_r 32
        --lora_alpha 16
        --report_to wandb
        --output_dir DeepSeek-R1-Distill-Qwen-1.5B-SFT

    resources:
      gpu: mi300x
      disk: 150GB
    ```
    </div>

=== "GRPO"

    <div editor-title="examples/llms/deepseek/trl/amd/grpo.dstack.yml">
    ```yaml
    type: task
    # The name is optional, if not specified, generated randomly
    name: trl-train-grpo

    image: rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0 

    # Required environment variables
    env:
      - WANDB_API_KEY
      - WANDB_PROJECT
      - MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # Commands of the task
    commands:
      - pip install trl
      - pip install datasets
      # numPy version less than 2 is required for the scipy installation with AMD.
      - pip install "numpy<2"
      - mkdir -p grpo_example
      - cp examples/llms/deepseek/trl/amd/grpo_train.py grpo_example/grpo_train.py
      - cd grpo_example
      - python grpo_train.py
        --model_name_or_path $MODEL_ID
        --dataset_name trl-lib/tldr
        --per_device_train_batch_size 2
        --logging_steps 25
        --output_dir Deepseek-Distill-Qwen-1.5B-GRPO
        --trust_remote_code

    # GRPO fine-tuning of DeepSeek-R1-Distill-Qwen-1.5B consumes 70% of VRAM
    resources:
      gpu: mi300x
      disk: 150GB
    ```
    </div>

### INTEL
Here is an example of LoRA fine-tuning of DeepSeek-R1-Distill-Qwen-7B on Intel Gaudi2 GPUs.

=== "LoRA"

    <div editor-title="examples/llms/deepseek/trl/intel/.dstack.yml">
    ```yaml
    type: task
    # The name is optional, if not specified, generated randomly
    name: trl-train

    image: vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0

    # Required environment variables
    env:
      - MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
      - WANDB_API_KEY
      - WANDB_PROJECT
    # Commands of the task
    commands:
      - pip install --upgrade-strategy eager optimum[habana]
      - pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
      - git clone https://github.com/huggingface/optimum-habana.git
      - cd optimum-habana/examples/trl
      - pip install -r requirements.txt
      - pip install wandb
      - DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 python ../gaudi_spawn.py --world_size 8 --use_deepspeed sft.py
        --model_name_or_path $MODEL_ID
        --dataset_name "lvwerra/stack-exchange-paired"
        --deepspeed ../language-modeling/llama2_ds_zero3_config.json
        --output_dir="./sft"
        --do_train
        --max_steps=500
        --logging_steps=10
        --save_steps=100
        --per_device_train_batch_size=1
        --per_device_eval_batch_size=1
        --gradient_accumulation_steps=2
        --learning_rate=1e-4
        --lr_scheduler_type="cosine"
        --warmup_steps=100
        --weight_decay=0.05
        --optim="paged_adamw_32bit"
        --lora_target_modules "q_proj" "v_proj"
        --bf16
        --remove_unused_columns=False
        --run_name="sft_deepseek_70"
        --report_to="wandb"
        --use_habana
        --use_lazy_mode

    resources:
      gpu: gaudi2:8

    ```
    </div>

### NVIDIA
Here are examples of LoRA fine-tuning of DeepSeek-R1-Distill-Qwen-1.5B and QLoRA fine-tuning of DeepSeek-V2-Lite on NVIDIA GPU.

=== "LoRA"

    <div editor-title="examples/llms/deepseek/trl/nvidia/.dstack.yml">
    ```yaml
    type: task
    # The name is optional, if not specified, generated randomly
    name: trl-train

    python: "3.10"

    # Required environment variables
    env:
      - WANDB_API_KEY
      - WANDB_PROJECT
      - MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # Commands of the task
    commands:
      - git clone https://github.com/huggingface/trl.git
      - pip install trl
      - pip install peft
      - pip install wandb
      - cd trl/trl/scripts
      - python sft.py
        --model_name_or_path $MODEL_ID
        --dataset_name trl-lib/Capybara
        --learning_rate 2.0e-4
        --num_train_epochs 1
        --packing
        --per_device_train_batch_size 2
        --gradient_accumulation_steps 8
        --gradient_checkpointing
        --logging_steps 25
        --eval_strategy steps
        --eval_steps 100
        --use_peft
        --lora_r 32
        --lora_alpha 16
        --report_to wandb
        --output_dir DeepSeek-R1-Distill-Qwen-1.5B-SFT

    resources:
      gpu: 24GB
    ```
    </div>

=== "QLoRA"

    <div editor-title="examples/llms/deepseek/trl/nvidia/deepseek_v2.dstack.yml">
    ```yaml
    type: task
    # The name is optional, if not specified, generated randomly
    name: trl-train-deepseek-v2

    python: "3.10"

    nvcc: true
    # Required environment variables
    env:
      - WANDB_API_KEY
      - WANDB_PROJECT
      - MODEL_ID=deepseek-ai/DeepSeek-V2-Lite
      - ACCELERATE_USE_FSDP=False
    # Commands of the task
    commands:
      - git clone https://github.com/huggingface/peft.git
      - pip install trl
      - pip install peft
      - pip install wandb
      - pip install bitsandbytes
      - cd peft/examples/sft
      - python train.py
        --seed 100 
        --model_name_or_path "deepseek-ai/DeepSeek-V2-Lite" 
        --dataset_name "smangrul/ultrachat-10k-chatml" 
        --chat_template_format "chatml" 
        --add_special_tokens False 
        --append_concat_token False 
        --splits "train,test" 
        --max_seq_len 512 
        --num_train_epochs 1 
        --logging_steps 5 
        --log_level "info" 
        --logging_strategy "steps" 
        --eval_strategy "epoch" 
        --save_strategy "epoch" 
        --hub_private_repo True 
        --hub_strategy "every_save" 
        --bf16 True 
        --packing True 
        --learning_rate 1e-4 
        --lr_scheduler_type "cosine" 
        --weight_decay 1e-4 
        --warmup_ratio 0.0 
        --max_grad_norm 1.0 
        --output_dir "mistral-sft-lora" 
        --per_device_train_batch_size 8 
        --per_device_eval_batch_size 8 
        --gradient_accumulation_steps 4 
        --gradient_checkpointing True 
        --use_reentrant True 
        --dataset_text_field "content" 
        --use_peft_lora True 
        --lora_r 16 
        --lora_alpha 16 
        --lora_dropout 0.05 
        --lora_target_modules "all-linear" 
        --use_4bit_quantization True 
        --use_nested_quant True 
        --bnb_4bit_compute_dtype "bfloat16"

    resources:
    # Consumes ~25GB of vRAM for QLoRA fine-tuning deepseek-ai/DeepSeek-V2-Lite
      gpu: 48GB
    ```

### Memory requirements

| Model Name               | Model size | Full fine-tuning | LoRA     | QLoRA    |
|--------------------------|------------|------------------|----------|----------|
|Deepseek-R1 (BF16)        | **671B**   | ~10.5TB          | ~1.4TB   | ~442GB   |
|DeepSeek-R1-Distill-Llama | **70B**    | ~1.09TB          | ~151GB   | ~46GB    |
|DeepSeek-R1-Distill-Qwen  | **32B**    | ~512GB           | ~70GB    | ~21GB    |
|DeepSeek-V2-Lite          | **16B**    | ~256GB           | ~35GB    | ~11GB    |
|DeepSeek-R1-Distill-Qwen  | **14B**    | ~224GB           | ~30GB    | ~9GB     |
|DeepSeek-R1-Distill-Llama | **8B**     | ~128GB           | ~17GB    | ~5GB     |
|DeepSeek-R1-Distill-Qwen	 | **7B**     | ~112GB           | ~15GB    | ~4GB     |
|DeepSeek-R1-Distill-Qwen	 | **1.5B**   | ~24GB            | ~3.2GB   | ~1GB     |

Note, The memory requirements are based on [`PyTorch's Finetuning LLMs Guide` :material-arrow-top-right-thin:{ .external }](https://pytorch.org/blog/finetune-llms/) and for LoRA and QLoRA we assume the size of the low-rank update matrices is 1% of the model parameters. But in practice a 7B model with QLoRA requires ~7-10GB due to intermediate hidden states.
Example calculations for 671B:
  - Full fine-tuning: 671B x 16Bytes = 10.48TB
  - LoRA: 671B x 2Bytes + 1% of 671B x 16Bytes = 1.41TB
  - QLoRA: 671B x 0.5Bytes + 1% of 671B x 16Bytes = 442GB


## Source code

The source-code of this example can be found in 
[`examples/llms/deepseek` :material-arrow-top-right-thin:{ .external }](https://github.com/dstackai/dstack/blob/master/examples/llms/deepseek).

## What's next?
1. Check [dev environments](https://dstack.ai/docs/dev-environments), [tasks](https://dstack.ai/docs/tasks), 
   [services](https://dstack.ai/docs/services), and [protips](https://dstack.ai/docs/protips).
2. Browse [AMD Instinct GPUs Power DeepSeek :material-arrow-top-right-thin:{ .external }](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), and [GRPO Training :material-arrow-top-right-thin:{ .external }](https://huggingface.co/docs/trl/main/en/grpo_trainer)

   
