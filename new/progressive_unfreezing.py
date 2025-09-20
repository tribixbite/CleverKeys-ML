#!/usr/bin/env python3
"""
Progressive Unfreezing Strategy for RNNT Training

Implements gradual unfreezing of model layers to preserve learned representations
while adapting to new data distributions (especially rare words).
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Set, Callable
import logging
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback


logger = logging.getLogger(__name__)


class ProgressiveUnfreezing:
    """
    Manages progressive unfreezing of model layers during training.

    Strategy:
    1. Start with frozen encoder (preserve common word knowledge)
    2. Gradually unfreeze encoder layers from top to bottom
    3. Keep decoder and joint network trainable throughout
    """

    def __init__(
        self,
        model: nn.Module,
        unfreeze_schedule: Optional[Dict[int, List[str]]] = None,
        initial_frozen: Optional[List[str]] = None,
        warmup_epochs: int = 2,
    ):
        """
        Initialize progressive unfreezing.

        Args:
            model: The RNNT model to apply unfreezing to
            unfreeze_schedule: Dict mapping epoch -> list of module names to unfreeze
            initial_frozen: List of module names to freeze initially
            warmup_epochs: Number of warmup epochs before starting unfreezing
        """
        self.model = model
        self.warmup_epochs = warmup_epochs

        # Default schedule if none provided
        if unfreeze_schedule is None:
            unfreeze_schedule = self._create_default_schedule()

        self.unfreeze_schedule = unfreeze_schedule

        # Default initial frozen layers
        if initial_frozen is None:
            initial_frozen = self._get_default_frozen_layers()

        self.initial_frozen = initial_frozen
        self.currently_frozen: Set[str] = set()

        # Initialize frozen state
        self._initialize_frozen_state()

    def _create_default_schedule(self) -> Dict[int, List[str]]:
        """Create default unfreezing schedule for RNNT model."""
        schedule = {
            # Warmup: Only decoder and joint are trainable
            0: [],  # Everything specified in initial_frozen is frozen

            # Start unfreezing encoder from top layers
            3: ['encoder.layers.11'],  # Top conformer layer
            4: ['encoder.layers.10'],
            5: ['encoder.layers.9'],
            6: ['encoder.layers.8'],
            7: ['encoder.layers.7'],
            8: ['encoder.layers.6'],
            9: ['encoder.layers.5'],
            10: ['encoder.layers.4'],
            11: ['encoder.layers.3'],
            12: ['encoder.layers.2'],
            13: ['encoder.layers.1'],
            14: ['encoder.layers.0'],  # Bottom conformer layer

            # Unfreeze preprocessing and embedding layers
            15: ['encoder.pre_encode', 'encoder.pos_encode'],

            # Full model unfrozen after epoch 15
        }
        return schedule

    def _get_default_frozen_layers(self) -> List[str]:
        """Get default list of layers to freeze initially."""
        # Freeze entire encoder initially
        return [
            'encoder',  # This will freeze all encoder submodules
        ]

    def _initialize_frozen_state(self):
        """Initialize the frozen state of the model."""
        logger.info("Initializing progressive unfreezing...")

        # Freeze specified modules
        for module_name in self.initial_frozen:
            self._freeze_module(module_name)

        self.currently_frozen = set(self.initial_frozen)

        # Log initial state
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Initial state: {trainable_params}/{total_params} "
            f"({100*trainable_params/total_params:.1f}%) parameters trainable"
        )

    def _freeze_module(self, module_name: str):
        """Freeze a module and all its submodules."""
        try:
            module = self._get_module_by_name(module_name)
            for param in module.parameters():
                param.requires_grad = False
            logger.debug(f"Froze module: {module_name}")
        except AttributeError:
            logger.warning(f"Module {module_name} not found in model")

    def _unfreeze_module(self, module_name: str):
        """Unfreeze a module and all its submodules."""
        try:
            module = self._get_module_by_name(module_name)
            for param in module.parameters():
                param.requires_grad = True
            logger.debug(f"Unfroze module: {module_name}")
        except AttributeError:
            logger.warning(f"Module {module_name} not found in model")

    def _get_module_by_name(self, module_name: str) -> nn.Module:
        """Get a module by its name path."""
        parts = module_name.split('.')
        module = self.model

        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        return module

    def step(self, epoch: int):
        """
        Update frozen/unfrozen state based on current epoch.

        Args:
            epoch: Current training epoch
        """
        if epoch < self.warmup_epochs:
            logger.info(f"Epoch {epoch}: Warmup phase, maintaining frozen state")
            return

        if epoch in self.unfreeze_schedule:
            modules_to_unfreeze = self.unfreeze_schedule[epoch]
            for module_name in modules_to_unfreeze:
                if module_name in self.currently_frozen:
                    self._unfreeze_module(module_name)
                    self.currently_frozen.remove(module_name)
                    logger.info(f"Epoch {epoch}: Unfroze {module_name}")

            # Log new state
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Epoch {epoch}: {trainable_params}/{total_params} "
                f"({100*trainable_params/total_params:.1f}%) parameters trainable"
            )

    def get_optimizer_param_groups(self, base_lr: float = 1e-4) -> List[Dict]:
        """
        Create optimizer parameter groups with different learning rates.

        Args:
            base_lr: Base learning rate for unfrozen parameters

        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []

        # Group 1: Decoder and Joint (always trainable, full LR)
        decoder_params = []
        joint_params = []

        for name, param in self.model.named_parameters():
            if 'decoder' in name:
                decoder_params.append(param)
            elif 'joint' in name:
                joint_params.append(param)

        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': base_lr,
                'name': 'decoder'
            })

        if joint_params:
            param_groups.append({
                'params': joint_params,
                'lr': base_lr,
                'name': 'joint'
            })

        # Group 2: Encoder layers (may be frozen, lower LR when unfrozen)
        encoder_params = []
        for name, param in self.model.named_parameters():
            if 'encoder' in name and param.requires_grad:
                encoder_params.append(param)

        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': base_lr * 0.1,  # Lower LR for encoder
                'name': 'encoder'
            })

        return param_groups


class ProgressiveUnfreezingCallback(Callback):
    """
    PyTorch Lightning callback for progressive unfreezing.
    """

    def __init__(
        self,
        unfreeze_schedule: Optional[Dict[int, List[str]]] = None,
        initial_frozen: Optional[List[str]] = None,
        warmup_epochs: int = 2,
    ):
        """
        Initialize callback.

        Args:
            unfreeze_schedule: Dict mapping epoch -> list of module names to unfreeze
            initial_frozen: List of module names to freeze initially
            warmup_epochs: Number of warmup epochs
        """
        super().__init__()
        self.unfreeze_schedule = unfreeze_schedule
        self.initial_frozen = initial_frozen
        self.warmup_epochs = warmup_epochs
        self.unfreezer: Optional[ProgressiveUnfreezing] = None

    def setup(self, trainer, pl_module: LightningModule, stage: str):
        """Setup unfreezing for the model."""
        if stage == 'fit':
            self.unfreezer = ProgressiveUnfreezing(
                model=pl_module,
                unfreeze_schedule=self.unfreeze_schedule,
                initial_frozen=self.initial_frozen,
                warmup_epochs=self.warmup_epochs,
            )

    def on_train_epoch_start(self, trainer, pl_module: LightningModule):
        """Called at the start of each training epoch."""
        if self.unfreezer:
            self.unfreezer.step(trainer.current_epoch)


class DiscriminativeLearningRates:
    """
    Implements discriminative learning rates for different layers.
    Lower layers get lower learning rates to preserve learned features.
    """

    def __init__(
        self,
        model: nn.Module,
        base_lr: float = 1e-4,
        lr_decay_factor: float = 0.95,
        min_lr_ratio: float = 0.01,
    ):
        """
        Initialize discriminative learning rates.

        Args:
            model: The model to apply discriminative LR to
            base_lr: Base learning rate for top layers
            lr_decay_factor: Factor to decay LR for each layer group
            min_lr_ratio: Minimum LR as ratio of base_lr
        """
        self.model = model
        self.base_lr = base_lr
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = base_lr * min_lr_ratio

    def get_layer_groups(self) -> List[List[nn.Parameter]]:
        """
        Group model parameters by layer depth.

        Returns:
            List of parameter groups from bottom to top layers
        """
        groups = []

        # For RNNT model structure
        # Group 1: Encoder preprocessing and embedding
        group1 = []
        # Group 2-13: Encoder conformer layers (12 layers)
        encoder_groups = [[] for _ in range(12)]
        # Group 14: Decoder
        decoder_group = []
        # Group 15: Joint network
        joint_group = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'encoder.pre_encode' in name or 'encoder.pos_encode' in name:
                group1.append(param)
            elif 'encoder.layers' in name:
                # Extract layer number
                try:
                    layer_idx = int(name.split('encoder.layers.')[1].split('.')[0])
                    encoder_groups[layer_idx].append(param)
                except (IndexError, ValueError):
                    group1.append(param)  # Fallback
            elif 'decoder' in name:
                decoder_group.append(param)
            elif 'joint' in name:
                joint_group.append(param)
            else:
                group1.append(param)  # Fallback for any other params

        # Combine groups (bottom to top)
        if group1:
            groups.append(group1)
        for group in encoder_groups:
            if group:
                groups.append(group)
        if decoder_group:
            groups.append(decoder_group)
        if joint_group:
            groups.append(joint_group)

        return groups

    def get_optimizer_param_groups(self) -> List[Dict]:
        """
        Create optimizer parameter groups with discriminative learning rates.

        Returns:
            List of parameter groups for optimizer
        """
        layer_groups = self.get_layer_groups()
        param_groups = []

        n_groups = len(layer_groups)
        for i, group in enumerate(layer_groups):
            if not group:
                continue

            # Calculate LR for this group (lower layers get lower LR)
            lr_multiplier = self.lr_decay_factor ** (n_groups - i - 1)
            group_lr = max(self.base_lr * lr_multiplier, self.min_lr)

            param_groups.append({
                'params': group,
                'lr': group_lr,
                'name': f'layer_group_{i}'
            })

            logger.debug(f"Layer group {i}: LR = {group_lr:.2e}")

        return param_groups


def create_unfreezing_schedule_for_profile(profile_name: str) -> Dict[int, List[str]]:
    """
    Create an unfreezing schedule tailored to a specific sampling profile.

    Args:
        profile_name: Name of the sampling profile

    Returns:
        Unfreezing schedule dictionary
    """
    if profile_name == 'rare_words':
        # Slower unfreezing for rare words training
        return {
            5: ['encoder.layers.11'],
            7: ['encoder.layers.10'],
            9: ['encoder.layers.9'],
            11: ['encoder.layers.8', 'encoder.layers.7'],
            13: ['encoder.layers.6', 'encoder.layers.5'],
            15: ['encoder.layers.4', 'encoder.layers.3'],
            17: ['encoder.layers.2', 'encoder.layers.1'],
            19: ['encoder.layers.0'],
            21: ['encoder.pre_encode', 'encoder.pos_encode'],
        }

    elif profile_name == 'long_words':
        # Focus on higher layers that capture longer dependencies
        return {
            3: ['encoder.layers.11', 'encoder.layers.10'],
            5: ['encoder.layers.9', 'encoder.layers.8'],
            7: ['encoder.layers.7', 'encoder.layers.6'],
            9: ['encoder.layers.5', 'encoder.layers.4'],
            11: ['encoder.layers.3', 'encoder.layers.2'],
            13: ['encoder.layers.1', 'encoder.layers.0'],
            15: ['encoder.pre_encode', 'encoder.pos_encode'],
        }

    else:
        # Default schedule
        return {
            3: ['encoder.layers.11'],
            4: ['encoder.layers.10'],
            5: ['encoder.layers.9'],
            6: ['encoder.layers.8'],
            7: ['encoder.layers.7'],
            8: ['encoder.layers.6'],
            9: ['encoder.layers.5'],
            10: ['encoder.layers.4'],
            11: ['encoder.layers.3'],
            12: ['encoder.layers.2'],
            13: ['encoder.layers.1'],
            14: ['encoder.layers.0'],
            15: ['encoder.pre_encode', 'encoder.pos_encode'],
        }