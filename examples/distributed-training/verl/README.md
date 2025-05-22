Submit Training Job
<div class="termy">

```shell
$ RAY_ADDRESS=http://localhost:8265
$ ray job submit \
  -- python3 -m verl.trainer.main_ppo \
     data.train_files=/root/data/gsm8k/train.parquet \
     data.val_files=/root/data/gsm8k/test.parquet \
     data.train_batch_size=256 \
     data.max_prompt_length=512 \
     data.max_response_length=256 \
     actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
     actor_rollout_ref.actor.optim.lr=1e-6 \
     actor_rollout_ref.actor.ppo_mini_batch_size=64 \
     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
     actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
     critic.optim.lr=1e-5 \
     critic.model.path=Qwen/Qwen2.5-7B-Instruct \
     critic.ppo_micro_batch_size_per_gpu=4 \
     algorithm.kl_ctrl.kl_coef=0.001 \
     trainer.project_name=ppo_training \
     trainer.experiment_name=qwen-2.5-7B \
     trainer.val_before_train=False \
     trainer.default_hdfs_dir=null \
     trainer.n_gpus_per_node=8 \
     trainer.nnodes=2 \
     trainer.default_local_dir=/checkpoints \
     trainer.save_freq=10 \
     trainer.test_freq=10 \
     trainer.total_epochs=15 2>&1 | tee verl_demo.log \
     trainer.resume_mode=disable
```
</div>

Internal Note: bash -c not needed here as `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` can be set in environment variable for `VERL` unlike `REGAN`
