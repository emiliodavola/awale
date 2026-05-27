07_training: Losses, batching, checkpointing, and reproducibility

Objective
- Train a shared policy/value network to minimize combined loss and produce robust game-playing policies.

Loss function
- For a batch of M examples {(s_i, π_i, z_i)}:
  - policy_loss = - (1/M) * sum_i sum_a π_i[a] * log_softmax(logits_i)[a]
  - value_loss = (1/M) * sum_i (v_i - z_i)^2  where v_i is scalar network value in [-1,1]
  - regularization = λ * sum(weights.^2)  (L2)
  - L = policy_loss + c_v * value_loss + regularization

Optimization
- Use Flux.Optimise with ADAM or SGD with momentum; ensure deterministic behavior by fixing seeds and disabling non-deterministic primitives.
- Support gradient clipping, learning rate scheduling, and checkpointed optimizer state

Batching
- Provide deterministic batch sampler: sample_batch(buffer, batch_size, rng)
- For multi-GPU later: design deterministic sharding strategy; for now support single-process batched training

Checkpointing and metadata
- Save: model parameters, optimizer state, RNG seed, training step, training config, and commit hash
- Checkpoints must be loadable to reproduce training continuation and evaluation

Evaluation during training
- Maintain a deterministic evaluation harness that runs fixed self-play matches and reports metrics (winrate, ELO estimate)

Contracts
- train_step(model, batch)::(model_new, loss_dict)
- save_checkpoint(path, model, optimizer, metadata)
- load_checkpoint(path)::(model, optimizer, metadata)

Testable properties
- Given a fixed sequence of minibatches and optimizer seed, train_step sequence is deterministic
- Checkpoint round-trip reproduces optimizer and model state to continue training with identical outcomes for same batches