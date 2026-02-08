"""
fsdp.py

Core class definition for a strategy implementing Torch native Fully Sharded Data Parallel Training (with support for
fine-grained control over wrapping policies and mixed precision per component).
"""
import gc
import json
import math
import threading
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# CPUOffload availability varies across PyTorch versions/installs; import safely.
try:
    from torch.distributed.fsdp.offload import CPUOffload
except Exception:
    try:
        from torch.distributed.fsdp import CPUOffload
    except Exception:
        CPUOffload = None
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from vitra.training.base_strategy import TrainingStrategy
from vitra.training.metrics import VLAMetrics
from vitra.utils.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)  


def get_constant_schedule_with_freeze_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a learning rate scheduler that is zero for the first `num_warmup_steps` steps, then constant."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return 0.0
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def split_modality_collator(
    vla,
    cognition_token_weight_decay: bool = False,
    move_word_embedding_to_action_model: bool = False,
    verbose: bool = True
):
    """
    Split model parameters into vlm backbone and other (action model) groups with separate decay settings.
    
    Returns:
        Tuple of (backbone_decay, backbone_no_decay, other_decay, other_no_decay) parameter lists
    """
    backbone_decay, backbone_no_decay, other_decay, other_no_decay = [], [], [], []
    
    def is_backbone_param(name: str) -> bool:
        """Check if the parameter is part of the vision or text backbone."""
        if move_word_embedding_to_action_model and "embed_tokens" in name:
            return False
        return "backbone" in name

    for name, param in vla.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check parameters that should not have weight decay
        no_weight_decay = param.ndim <= 1 or name.endswith(".bias")
        if "cognition_token" in name:
            no_weight_decay = not cognition_token_weight_decay
        
        # Categorize parameters
        if no_weight_decay:
            if is_backbone_param(name):
                backbone_no_decay.append(param)
                if verbose:
                    overwatch.info(f"Parameter `{name}` is part of the backbone and has no decay; added to `backbone_no_decay`")
            else:
                other_no_decay.append(param)
                if verbose:
                    overwatch.info(f"Parameter `{name}` is not part of the backbone and has no decay; added to `other_no_decay`")
        else:
            if is_backbone_param(name):
                backbone_decay.append(param)
                if verbose:
                    overwatch.info(f"Parameter `{name}` is part of the backbone and has decay; added to `backbone_decay`")
            else:
                other_decay.append(param)
                if verbose:
                    overwatch.info(f"Parameter `{name}` is not part of the backbone and has decay; added to `other_decay`")
    
    return backbone_decay, backbone_no_decay, other_decay, other_no_decay


class VLAFSDPStrategy(TrainingStrategy):
    """FSDP (Fully Sharded Data Parallel) training strategy for VLA models."""

    def __init__(
        self,
        vla,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        action_model_learning_rate: Optional[float] = None,
        action_model_weight_decay: Optional[float] = None,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        sharding_strategy: str = "shard-grad-op",
        state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
        cognition_token_weight_decay: bool = False,
        llm_freeze_step: int = 0,
        move_word_embedding_to_action_model: bool = False,
        optimizer_betas: tuple = (0.9, 0.999),
    ) -> None:
        super().__init__(
            vla=vla,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
        )
        # Action model specific parameters
        self.action_model_learning_rate = action_model_learning_rate if action_model_learning_rate is not None else learning_rate
        self.action_model_weight_decay = action_model_weight_decay if action_model_weight_decay is not None else weight_decay
        self.cognition_token_weight_decay = cognition_token_weight_decay
        self.llm_freeze_step = llm_freeze_step
        self.move_word_embedding_to_action_model = move_word_embedding_to_action_model
        self.optimizer_betas = optimizer_betas

        # FSDP-specific parameters
        if sharding_strategy == "shard-grad-op":
            self.fsdp_sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        elif sharding_strategy == "full-shard":
            self.fsdp_sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            raise ValueError(f"FSDP sharding strategy '{sharding_strategy}' is not supported!")

        assert state_dict_type == StateDictType.FULL_STATE_DICT, "Sharded state saving is not yet implemented!"
        self.fsdp_state_dict_type = state_dict_type
        self.fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        self.fsdp_save_optimizer_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        only_trainable: bool = True,
        is_epoch_end: bool = False,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vla, FSDP), "FSDPStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!"
        if is_epoch_end:
            checkpoint_name = f"epoch={epoch}-step={global_step}.end.ckpt"
        else:
            checkpoint_name = f"epoch={epoch}-step={global_step}.ckpt"
        checkpoint_dir = run_dir / "checkpoints"/ checkpoint_name
        if overwatch.is_rank_zero():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        def save_with_time(state_dict, path):
            overwatch.info(f"Saving state dict to {path} start at {datetime.now()}")
            torch.save(state_dict, path)
            overwatch.info(f"Saving state dict to {path} end at {datetime.now()}")
        
        # Gather full state dictionary from shards
        with FSDP.state_dict_type(self.vla, self.fsdp_state_dict_type, self.fsdp_save_policy, self.fsdp_save_optimizer_policy):
            overwatch.info("Gathering model state")
            model_state = self.vla.state_dict()
            overwatch.info("Preparing save checkpoint")
            overwatch.info("Gathering optimizer state")
            optim_state = FSDP.optim_state_dict(self.vla, self.optimizer)
            meta_state = {
                "epoch": epoch,
                "global_step": global_step
            }
            if overwatch.is_rank_zero():
                with open(checkpoint_dir / "meta.json", "w") as f:
                    json.dump(meta_state, f)
            dist.barrier()
            if overwatch.is_rank_zero():
                threading.Thread(target=save_with_time, args=(model_state, checkpoint_dir / 'weights.pt')).start()
                threading.Thread(target=save_with_time, args=(optim_state, checkpoint_dir / 'optimizer.pt')).start()
            
            dist.barrier()

    def load_optimizer_and_scheduler(self, checkpoint_folder: str) -> None:
        """Load optimizer and scheduler state from checkpoint."""
        assert isinstance(self.vla, FSDP), "FSDPStrategy.load_optimizer_and_scheduler assumes VLM is already wrapped in FSDP!"
        
        checkpoint_folder = Path(checkpoint_folder)
        optimizer_path = checkpoint_folder / "optimizer.pt"
        
        if not optimizer_path.exists():
            overwatch.warning(f"Optimizer checkpoint not found at {optimizer_path}!")
            return
        
        # Load checkpoint (FSDP handles device placement automatically)
        optim_state_dict = torch.load(optimizer_path, map_location="cpu")
        
        with FSDP.state_dict_type(
            self.vla,
            self.fsdp_state_dict_type,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        ):
            optim_state_dict = FSDP.optim_state_dict_to_load(self.vla, self.optimizer, optim_state_dict)
            # optim_state_dict = FSDP.optim_state_dict_to_load(self.vla, self.optimizer, optim_state_dict["optimizer"])
            self.optimizer.load_state_dict(optim_state_dict)
        
        overwatch.info(f"Loaded optimizer state dict from {optimizer_path}")
        
    def run_setup(
        self,
        run_dir: Path,
        n_train_examples: int,
        auto_wrap_policy_modules,
        checkpointing_policy_modules,
    ) -> None:
        """Setup FSDP training (wrap model, create optimizer, etc.)."""
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        auto_wrap_policy = ModuleWrapPolicy(auto_wrap_policy_modules)

        # Configure FSDP mixed precision policy
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_buffer_dtype,
                buffer_dtype=reduce_buffer_dtype
            )
        else:
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32
            )

        # <FSDP> => note that FSDP will automatically take care of device placement (similar to `autocast`)
        # Configure optional CPU offload only when `CPUOffload` is available in this torch build.
        cpu_offload_arg = CPUOffload(offload_params=True) if CPUOffload is not None else None
        fsdp_kwargs = dict(
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=self.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=False,
        )
        if cpu_offload_arg is not None:
            fsdp_kwargs["cpu_offload"] = cpu_offload_arg

        self.vla = FSDP(self.vla, **fsdp_kwargs)
        
        # Setup gradient checkpointing
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing under FSDP --> we make the same assumption as in the DDP/other strategies; the
            #   bulk of activation memory is taken up by the LLM activations. However, unlike other strategies, we
            #   cannot rely on the HF Transformers default `gradient_checkpointing_enable()` --> FSDP breaks semantics!
            #
            # Instead, we need to write our own *NO-REENTRANT* wrapper, and apply it to the LLM's Transformer Layer.
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            if checkpointing_policy_modules is not None:
                def check_fn(submodule: nn.Module) -> bool:
                    if isinstance(checkpointing_policy_modules, (list, set)):
                        return any(isinstance(submodule, module) for module in checkpointing_policy_modules)
                    return isinstance(submodule, checkpointing_policy_modules)

                # Note that the terms "activation checkpointing" and "gradient checkpointing" are synonymous!
                apply_activation_checkpointing(self.vla, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

        dist.barrier()

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        backbone_decay, backbone_no_decay, other_decay, other_no_decay = split_modality_collator(
            self.vla,
            cognition_token_weight_decay=self.cognition_token_weight_decay,
            move_word_embedding_to_action_model=self.move_word_embedding_to_action_model,
            verbose=False
        )
        groups = [
            {"params": backbone_decay, "weight_decay": self.weight_decay, "lr": self.learning_rate},
            {"params": backbone_no_decay, "weight_decay": 0.0, "lr": self.learning_rate},
            {"params": other_decay, "weight_decay": self.action_model_weight_decay, "lr": self.action_model_learning_rate},
            {"params": other_no_decay, "weight_decay": 0.0, "lr": self.action_model_learning_rate},
        ]

        # Create Optimizer & LR Scheduler
        self.optimizer = AdamW(groups, betas=self.optimizer_betas)

        if self.lr_scheduler_type == "linear-warmup+cosine-decay" or self.lr_scheduler_type == "warmup_cosine":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            self.lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
                num_training_steps,
                min_lr_rate=0.1
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        elif self.lr_scheduler_type == "linear-warmup+constant" or self.lr_scheduler_type == "warmup_constant":
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_warmup_steps
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "backbone-freeze-warmup":
            # Backbone uses constant-freeze-warmup, action head uses constant
            num_warmup_steps = self.llm_freeze_step

            # Create separate optimizers for different scheduling
            backbone_groups = [
                {"params": backbone_decay, "weight_decay": self.weight_decay, "lr": self.learning_rate},
                {"params": backbone_no_decay, "weight_decay": 0.0, "lr": self.learning_rate},
            ]
            action_model_groups = [
                {"params": other_decay, "weight_decay": self.action_model_weight_decay, "lr": self.action_model_learning_rate},
                {"params": other_no_decay, "weight_decay": 0.0, "lr": self.action_model_learning_rate},
            ]

            # Create separate optimizers for backbone and action model
            backbone_optimizer = AdamW(backbone_groups, betas=self.optimizer_betas)
            action_model_optimizer = AdamW(action_model_groups, betas=self.optimizer_betas)
            
            # Create schedulers for each component
            backbone_scheduler = get_constant_schedule_with_freeze_warmup(
                backbone_optimizer, num_warmup_steps=num_warmup_steps
            )
            action_model_scheduler = get_constant_schedule(action_model_optimizer)
            
            # Create the multi-group scheduler
            self.lr_scheduler = MultiGroupLRScheduler(
                self.optimizer, backbone_scheduler, action_model_scheduler
            )
        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log!
        scheduler_info = f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
        if self.lr_scheduler_type == "backbone-freeze-warmup+action-constant":
            scheduler_info += f"                 |-> Backbone: Constant schedule with freeze warmup ({num_warmup_steps} steps)\n"
            scheduler_info += f"                 |-> Action Head: Constant schedule\n"
        else:
            scheduler_info += f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"

        overwatch.info(
            "FSDP Full-Shard Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n"
            f"                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n"
            f"                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> AdamW Betas = {self.optimizer_betas}\n"
            + scheduler_info +
            f"         |-> LLM Learning Rate = {self.learning_rate}\n"
            f"         |-> Action Model Learning Rate = {self.action_model_learning_rate}\n"
            f"         |-> LLM Weight Decay = {self.weight_decay}\n"
            f"         |-> Action Model Weight Decay = {self.action_model_weight_decay}\n"
            f"         |-> Cognition Token Weight Decay = {self.cognition_token_weight_decay}\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        """Clip gradients using FSDP's built-in gradient clipping."""
        self.vla.clip_grad_norm_(max_norm=self.max_grad_norm)

    def run_training(
        self,
        dataloader,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        epoch_save_interval: int = 1,
        start_epoch: int = 0,
        start_global_step: int = 0,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given dataloader; log losses and action metrics to metrics."""
        vla_dataset = dataloader.dataset
        
        # # Inspect the DataLoader
        # print(f"DataLoader: {dataloader}")                      # <torch.utils.data.dataloader.DataLoader object at 0x77ba1ebbbfa0>
        # print(f"Dataset type: {type(dataloader.dataset)}")      # <class 'vitra.datasets.dataset.MultipleWeightedDataset'>
        # print(f"Dataset length: {len(dataloader.dataset)}")     # 1125013
        # print(f"Batch sampler: {dataloader.batch_sampler}")     # <vitra.datasets.dataset.MultipleDatasetWeightedDistributedBatchSampler object at 0x77b903b9f8e0>
        # print(f"Collate function: {dataloader.collate_fn}")     # PaddedCollatorForHandPrediction(model_max_length=1000000000000000019884624838656, pad_token_id=0, padding_side='right', pixel_values_dtype=torch.float32)

        # # If it's a MultipleWeightedDataset, inspect sub-datasets
        # if hasattr(dataloader.dataset, 'datasets'):
        #     print(f"Number of sub-datasets: {len(dataloader.dataset.datasets)}")
        #     for i, ds in enumerate(dataloader.dataset.datasets):
        #         print(f"  Sub-dataset {i}: {type(ds).__name__} with {len(ds)} samples")

        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * (len(dataloader) // self.grad_accumulation_steps)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
            initial=start_global_step,
        ) as progress:
            train_idx = 0
            for epoch in range(start_epoch, self.epochs):
                self.vla.train()
                self.optimizer.zero_grad()

                # === DataLoader Iteration Flow ===
                # When we call `for batch_idx, batch in enumerate(dataloader)`, the following happens:
                #
                # 1. DataLoader.__iter__() is invoked, starting the iteration
                #
                # 2. batch_sampler yields a list of sample indices for the current batch
                #    Example: [(0, 5), (1, 10), (0, 15), ..., (2, 42)]  # (dataset_id, sample_idx) tuples
                #    - Created by: MultipleDatasetWeightedDistributedBatchSampler (in dataset.py)
                #    - This sampler handles weighted sampling across multiple datasets with distributed training
                #
                # 3. For EACH index in the batch, DataLoader calls: vla_dataset.__getitem__(index)
                #    - This happens per_device_batch_size times (e.g., 8 times for batch_size=8)
                #
                # 4. MultipleWeightedDataset.__getitem__(index) receives the (dataset_id, idx) tuple:
                #    - Determines which sub-dataset to use based on dataset_id
                #    - Calls: self.datasets[dataset_id][idx]
                #
                # 5. FrameDataset.__getitem__(idx) is invoked:
                #    - Calls: self.episodic_dataset_core.__getitem__(idx)
                #    - This returns raw episode data (images, hand poses, actions, etc.)
                #
                # 6. EpisodicDatasetCore.__getitem__(idx) (in human_dataset.py):
                #    - Loads episode annotation from .npy file
                #    - Extracts frame indices, hand poses, actions, camera params
                #    - Loads video frames using cv2.VideoCapture
                #    - Returns dict with: image_list, action_list, action_mask, current_state, etc.
                #
                # 7. FrameDataset.post_transform(sample) is called:
                #    - Converts images to PIL format
                #    - Applies processor (PaliGemma) to tokenize text and process images
                #    - Returns dict with keys: pixel_values, input_ids, labels, actions, action_masks,
                #      current_state, current_state_mask, fov, intrinsics, dataset_name
                #
                # 8. DataLoader collects all individual dicts into a list: instances = [dict₁, dict₂, ..., dict₈]
                #
                # 9. DataLoader calls the collator: collator(instances)
                #    - Invokes: PaddedCollatorForHandPrediction.__call__(self, instances)
                #
                # 10. PaddedCollatorForHandPrediction processes the instances list:
                #     - Extracts lists: input_ids=[seq₁, seq₂, ...], pixel_values=[img₁, img₂, ...], etc.
                #     - Pads input_ids and labels to same length using pad_sequence()
                #     - Stacks tensors: torch.stack(fov), torch.stack(intrinsics), torch.stack(actions), etc.
                #     - Creates attention_mask based on pad token positions
                #     - Reshapes pixel_values to flatten batch and sequence dimensions
                #     - Returns batched dict with all tensors properly shaped for model input
                #
                # 11. The `batch` variable now contains the collated, batched dictionary with keys:
                #     - pixel_values: [batch_size*seq_len, 3, H, W] - flattened images ready for vision encoder
                #     - input_ids: [batch_size, max_seq_len] - padded token sequences
                #     - attention_mask: [batch_size, max_seq_len] - mask for valid (non-padded) tokens
                #     - actions: [batch_size, action_dim] - hand action targets (e.g., 102-dim for both hands)
                #     - action_masks: [batch_size, action_dim] - mask for valid action dimensions
                #     - current_state: [batch_size, state_dim] - current hand state
                #     - current_state_mask: [batch_size, state_dim] - mask for current state
                #     - fov: [batch_size, fov_dim] - field of view parameters
                #     - intrinsics: [batch_size, intrinsic_dim] - camera intrinsic matrices
                #     - dataset_names: List[str] - optional dataset identifiers for each sample
                #

                # it first enters file:///home/haziq/VITRA/vitra/datasets/dataset.py FrameDataset.__getitem__()
                for batch_idx, batch in enumerate(dataloader):
                    # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                    #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                    
                    # === Unpack Batch Variables ===
                    # All tensors below come from PaddedCollatorForHandPrediction's output dictionary
                    
                    # input_ids: [batch_size, max_seq_len] - Tokenized text input (language instructions)
                    #   - Contains token IDs from the PaliGemma processor/tokenizer
                    #   - Padded sequences of text like "Pick up the cup" converted to token IDs
                    #   - Example shape: [8, 256] for batch_size=8, max_seq_len=256
                    #   - Includes special tokens: <image> tokens, instruction tokens, padding tokens
                    input_ids = batch["input_ids"]
                    
                    # rgb (pixel_values): [batch_size*seq_len, 3, H, W] - Video frames/images
                    #   - RGB image tensors preprocessed by PaliGemma's image processor
                    #   - Already normalized and resized (typically H=W=224 for PaliGemma)
                    #   - Flattened from [batch_size, seq_len, 3, H, W] to combine batch and temporal dims
                    #   - Example shape: [8, 3, 224, 224] if seq_len=1 image per sample
                    #   - Contains egocentric hand manipulation video frames
                    rgb = batch["pixel_values"]
                    
                    # attention_mask: [batch_size, max_seq_len] - Mask for valid tokens vs padding
                    #   - Boolean/binary mask: 1 = valid token, 0 = padding token
                    #   - Used to ignore padding tokens during attention computation in transformer
                    #   - Example: [1,1,1,1,1,0,0,0] means first 5 tokens are valid, last 3 are padding
                    #   - Ensures model doesn't attend to meaningless padding tokens
                    #   - IMPORTANT: For PaliGemma with images, most tokens are IMAGE TOKENS (e.g., token_id=257152)
                    #     * PaliGemma vision encoder outputs ~256 image patch tokens per image
                    #     * These image tokens come BEFORE the text tokens in the sequence
                    #     * Example input_ids: [257152, 257152, ..., 257152, 2, 7850, 1634, ...]
                    #       where 257152 = <image> tokens, and [2, 7850, ...] = actual text tokens
                    #     * If no padding needed, attention_mask will be all True (all tokens valid)
                    #     * Padding only occurs when text sequences have different lengths
                    attention_mask = batch["attention_mask"]
                    
                    # action_labels: [batch_size, action_dim] - Ground truth hand actions to predict
                    #   - For VITRA: action_dim = 102 (51 per hand × 2 hands)
                    #   - Each hand action (51-dim): [tx, ty, tz, rx, ry, rz, joint1_x, joint1_y, joint1_z, ...]
                    #       * tx, ty, tz (3): wrist translation in camera space (meters)
                    #       * rx, ry, rz (3): wrist rotation as Euler angles (radians)
                    #       * 45 joint angles (15 joints × 3 Euler angles): MANO hand pose parameters
                    #   - Left hand (0:51) + Right hand (51:102)
                    #   - These are the target values the model learns to predict
                    #   - Normalized using dataset statistics (mean/std normalization)
                    #
                    #   - IMPORTANT: This is NOT iterative/autoregressive prediction!
                    #     * Model predicts actions for a CHUNK of future timesteps simultaneously
                    #     * chunk_size (e.g., 16) = number of future frames to predict
                    #     * action_labels shape: [batch_size, chunk_size, 102] in training
                    #     * During inference, model outputs all 16 future actions at once
                    #     * Example: Given current frame, predict hand movements for next 16 frames
                    #     * This is "open-loop" prediction, not "closed-loop" (no feedback between timesteps)
                    #     * Diffusion model generates entire action trajectory in one forward pass
                    action_labels = batch["actions"]
                    
                    # action_masks: [batch_size, action_dim] or [batch_size, 2] - Validity mask for actions
                    #   - Indicates which hand actions are valid/visible in this frame
                    #   - Shape [batch_size, 2]: one mask per hand [left_valid, right_valid]
                    #   - True/1 = hand is visible and action is valid, False/0 = hand occluded/missing
                    #   - Used to exclude loss computation for invalid/missing hand data
                    #   - Example: [True, False] means left hand visible, right hand not visible
                    action_masks = batch["action_masks"]
                    
                    # current_state_mask: [batch_size, 2] - Mask for current hand state availability
                    #   - Boolean mask indicating if current hand state is available for each hand
                    #   - [left_hand_available, right_hand_available]
                    #   - True = hand state is available for conditioning, False = no state info
                    #   - Can be intentionally masked during training (with state_mask_prob) for robustness
                    #   - When False, model must predict without knowing current hand configuration
                    current_state_mask = batch["current_state_mask"]
                    
                    # current_state: [batch_size, state_dim] - Current hand state/configuration
                    #   - For VITRA: state_dim = 122 (61 per hand × 2 hands)
                    #   - Each hand state (61-dim): [tx, ty, tz, rx, ry, rz, joint_poses(45), betas(10)]
                    #       * Translation (3) + Rotation (3) + MANO pose (45) + MANO shape params (10)
                    #       * betas are MANO shape parameters (hand shape variations like finger thickness)
                    #   - Left hand (0:61) + Right hand (61:122)
                    #   - Provides the starting hand configuration for action prediction
                    #   - Set to zeros when current_state_mask is False
                    #   - Normalized using dataset statistics
                    current_state = batch["current_state"]
                    
                    # fov: [batch_size, 2] - Camera field of view parameters
                    #   - [horizontal_fov, vertical_fov] in radians
                    #   - Calculated from camera intrinsics: fov = 2 * arctan(image_dim / (2 * focal_length))
                    #   - Used for 3D spatial reasoning and geometric consistency
                    #   - Important for projecting 3D hand poses to 2D image space
                    #   - Example: [1.2, 0.9] radians ≈ [69°, 52°] FOV
                    #   - Varies by dataset (EgoExo4D fisheye vs Ego4D pinhole cameras)
                    fov = batch["fov"]
                    
                    # Additional available fields (not unpacked here but in batch dict):
                    # - intrinsics: [batch_size, 3, 3] - Camera intrinsic matrix (focal lengths, principal point)
                    # - labels: [batch_size, max_seq_len] - Typically unused for hand prediction (kept for compatibility)
                    # - dataset_names: List[str] - Source dataset identifier (e.g., "ego4d_cooking", "ssv2")
                    
                    # === What are we predicting? ===
                    # This is NOT language modeling or PaliGemma pretraining!
                    # This is Vision-Language-Action (VLA) training for hand manipulation:
                    #
                    # INPUTS TO MODEL:
                    #   1. rgb (images): Video frames showing hands manipulating objects
                    #   2. input_ids: Tokenized instruction text (e.g., "Pick up the cup")
                    #   3. current_state: Current hand configuration (optional, can be masked)
                    #   4. fov: Camera field of view parameters
                    #
                    # OUTPUT FROM MODEL (what we predict):
                    #   - action_labels: Future hand actions [batch_size, 102] 
                    #     * 102 dims = 51 per hand × 2 hands
                    #     * Each hand: [translation(3) + rotation(3) + joint_poses(45)]
                    #     * Model learns to predict: "Given this image and instruction, what should the hands do next?"
                    #
                    # HOW PREDICTION WORKS:
                    #   1. PaliGemma backbone processes: images + text + cognition_token
                    #      - Vision encoder: extracts visual features from rgb frames
                    #      - Language encoder: processes instruction text (input_ids)
                    #      - attention_mask: tells transformer which tokens are valid (not padding)
                    #      - Special "cognition_token": a learned token that aggregates multimodal info
                    #
                    #   2. extract_cognition_token(): Extracts the last valid token's hidden state
                    #      - Uses attention_mask to find last non-padded position
                    #      - This token embedding captures: visual scene + language instruction + context
                    #      - Shape: [batch_size, hidden_dim] (e.g., [8, 2048])
                    #
                    #   3. DiT action model: Diffusion Transformer predicts hand actions
                    #      - Takes cognition_token features as conditioning
                    #      - Optionally uses current_state for better predictions
                    #      - Outputs 102-dim action predictions via denoising diffusion
                    #
                    # LOSS COMPUTATION:
                    #   - Diffusion loss: MSE between predicted noise and actual noise
                    #   - Only computed on valid hands (using action_masks)
                    #   - Trains model to denoise random noise into coherent hand actions
                    #
                    # THIS IS FINE-TUNING, NOT PRETRAINING:
                    #   - PaliGemma backbone starts from pretrained vision-language weights
                    #   - We freeze most of PaliGemma and only train:
                    #     * cognition_token embedding (new learnable token)
                    #     * DiT action model (new diffusion head)
                    #     * Optionally: last few layers of PaliGemma for better feature adaptation
                    #   - Goal: Adapt vision-language model to predict robotic/hand actions
                    #
                    # INTUITION:
                    #   Model learns: "If I see this scene and hear this instruction, 
                    #                  what hand movements would accomplish the task?"
                    #   Example: Image shows cup, instruction "pick up the cup"
                    #            → Model predicts: reach toward cup, grasp, lift upward

                    # Run forward under AMP autocast when mixed precision is enabled so
                    # tensors are cast consistently (avoids Float vs BFloat16 dtype errors).
                    if self.enable_mixed_precision_training:
                        with autocast(dtype=self.mixed_precision_dtype):
                            prediction = self.vla.forward(
                                rgb,
                                input_ids,
                                attention_mask=attention_mask,
                                action_labels=action_labels,
                                action_masks=action_masks,
                                current_state_mask=current_state_mask,
                                current_state=current_state,
                                data_source=['action'],
                                fov=fov,
                            )
                    else:
                        prediction = self.vla.forward(
                            rgb,
                            input_ids,
                            attention_mask=attention_mask,
                            action_labels=action_labels,
                            action_masks=action_masks,
                            current_state_mask=current_state_mask,
                            current_state=current_state,
                            data_source=['action'],
                            fov=fov,
                        )
                    loss = prediction["loss"]

                    # Commit loss and backward
                    metrics.commit(
                        loss=loss, 
                        left_hand_6d=prediction["left_hand_6d"],
                        left_hand_joints=prediction["left_hand_joints"],
                        right_hand_6d=prediction["right_hand_6d"],
                        right_hand_joints=prediction["right_hand_joints"],
                    )
                    
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # === Gradient Step ===
                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()

                        self.optimizer.zero_grad()
                        # Compute epoch value using number of completed gradient steps
                        # epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                        # Prepare learning rate metrics
                        lr_dict = {}
                        # Get the appropriate learning rate for logging
                        if isinstance(self.lr_scheduler, MultiGroupLRScheduler):
                            # For multi-group scheduler, log multiple learning rates
                            lr = self.lr_scheduler.get_last_lr()
                            lr_dict['backbone_decay_lr'] = lr[0]       # backbone decay learning rate
                            lr_dict['backbone_no_decay_lr'] = lr[1]    # backbone no decay learning rate
                            lr_dict['action_decay_lr'] = lr[2]         # action decay learning rate
                            lr_dict['action_no_decay_lr'] = lr[3]      # action no decay learning rate
                            current_lr = lr_dict['backbone_decay_lr']  # backbone learning rate
                        else:
                            current_lr = self.lr_scheduler.get_last_lr()[0]
                        
                        metrics.commit(update_step_time=True, global_step=metrics.global_step + 1, epoch=epoch, lr=current_lr, **lr_dict)
                        status = metrics.push()

                        # Check for Save Interval or Max Steps & Save Checkpoint
                        if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                            (metrics.global_step % save_interval) == 0
                        ):
                            self.save_checkpoint(
                                metrics.run_dir, metrics.global_step, epoch, only_trainable=not save_full_model
                            )
                            dist.barrier()

                        if terminate:
                            return
                    train_idx += 1

                    # Update progress bar
                    progress.set_description(status)
                    progress.update()
                    
                # Save epoch checkpoint if needed
                if epoch % epoch_save_interval == 0:
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, only_trainable=not save_full_model, is_epoch_end=True
                    )
                gc.collect()
                torch.cuda.empty_cache()

# Custom LR Scheduler for different parameter groups
class MultiGroupLRScheduler:
    """
    A custom learning rate scheduler that applies different scheduling strategies
    to different parameter groups in the optimizer.
    """
    def __init__(self, optimizer, backbone_scheduler, action_model_scheduler):
        self.optimizer = optimizer
        self.backbone_scheduler = backbone_scheduler
        self.action_model_scheduler = action_model_scheduler
        
        # Assume first two groups are backbone (decay/no_decay), last two are action model
        self.backbone_group_indices = [0, 1]
        self.action_model_group_indices = [2, 3]
    
    def step(self):
        """Step both schedulers and update the corresponding parameter groups"""
        # Step the schedulers
        self.backbone_scheduler.step()
        self.action_model_scheduler.step()
        
        # Update backbone parameter groups with backbone scheduler's learning rates
        backbone_lrs = self.backbone_scheduler.get_last_lr()
        for i, group_idx in enumerate(self.backbone_group_indices):
            # Both backbone groups should use the same LR from backbone scheduler
            self.optimizer.param_groups[group_idx]['lr'] = backbone_lrs[0] if len(backbone_lrs) == 1 else backbone_lrs[i]
        
        # Update action model parameter groups with action model scheduler's learning rates
        action_model_lrs = self.action_model_scheduler.get_last_lr()
        for i, group_idx in enumerate(self.action_model_group_indices):
            # Both action model groups should use the same LR from action model scheduler
            self.optimizer.param_groups[group_idx]['lr'] = action_model_lrs[0] if len(action_model_lrs) == 1 else action_model_lrs[i]
    
    def get_last_lr(self):
        """Return the last learning rates for all parameter groups"""
        backbone_lrs = self.backbone_scheduler.get_last_lr()
        action_model_lrs = self.action_model_scheduler.get_last_lr()
        
        # Return LRs in the order of parameter groups: [backbone_decay, backbone_no_decay, action_decay, action_no_decay]
        return [
            backbone_lrs[0] if len(backbone_lrs) == 1 else backbone_lrs[0],  # backbone_decay
            backbone_lrs[0] if len(backbone_lrs) == 1 else backbone_lrs[0],  # backbone_no_decay
            action_model_lrs[0] if len(action_model_lrs) == 1 else action_model_lrs[0],  # action_decay
            action_model_lrs[0] if len(action_model_lrs) == 1 else action_model_lrs[0],  # action_no_decay
        ]
