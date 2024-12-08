import logging
import os
from functools import wraps
from typing import Callable, Literal, overload

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, is_torch_npu_available
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import PushToHubMixin

from sentence_transformers import CrossEncoder
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import fullname, get_device_name, import_from_string

import copy
import importlib
import json
import logging
import math
import os
import queue
import shutil
import sys
import tempfile
import traceback
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Callable, Literal, overload

import numpy as np
import torch
import torch.multiprocessing as mp
import transformers
from huggingface_hub import HfApi
from numpy import ndarray
from torch import Tensor, device, nn
from tqdm.autonotebook import trange
from transformers import is_torch_npu_available
from transformers.dynamic_module_utils import get_class_from_dynamic_module, get_relative_import_files

from sentence_transformers.model_card import SentenceTransformerModelCardData, generate_model_card
from sentence_transformers.similarity_functions import SimilarityFunction
logger = logging.getLogger(__name__)


class CrossEncoderCL(CrossEncoder):
    def smart_batching_collate(self, batch: list[InputExample]) -> tuple[BatchEncoding, Tensor]:
        # parity check
        try:
            for i in range(0, len(batch), 4):
                parity = batch[i:i+4]
                if not (parity[0][1] == parity[1][1] and parity[1][1] == parity[2][1] and parity[2][1] == parity[3][1]):
                    raise RuntimeError("parity check failed")
                batch[i] = batch[i][0]
                batch[i+1] = batch[i+1][0]
                batch[i+2] = batch[i+2][0]
                batch[i+3] = batch[i+3][0]
                
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print("parity check failed")     
            raise RuntimeError("parity check failed")   
        
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for i, example in enumerate(batch):
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            if i % 4 == 0: labels.append(example.label)
        
        tokenized = self.tokenizer(
            *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.max_length
        )
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(
            self.model.device
        )

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.model.device)

        return tokenized, labels
    
    def fit(
        self,
        train_dataloader: DataLoader,
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        loss_fct=None,
        activation_fct=nn.Identity(),
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: type[Optimizer] = torch.optim.AdamW,
        optimizer_params: dict[str, object] = {"lr": 1e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        Args:
            train_dataloader (DataLoader): DataLoader with training InputExamples
            evaluator (SentenceEvaluator, optional): An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc. Defaults to None.
            epochs (int, optional): Number of epochs for training. Defaults to 1.
            loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss(). Defaults to None.
            activation_fct: Activation function applied on top of logits output of model.
            scheduler (str, optional): Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts. Defaults to "WarmupLinear".
            warmup_steps (int, optional): Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero. Defaults to 10000.
            optimizer_class (Type[Optimizer], optional): Optimizer. Defaults to torch.optim.AdamW.
            optimizer_params (Dict[str, object], optional): Optimizer parameters. Defaults to {"lr": 2e-5}.
            weight_decay (float, optional): Weight decay for model parameters. Defaults to 0.01.
            evaluation_steps (int, optional): If > 0, evaluate the model using evaluator after each number of training steps. Defaults to 0.
            output_path (str, optional): Storage path for the model and evaluation files. Defaults to None.
            save_best_model (bool, optional): If true, the best model (according to evaluator) is stored at output_path. Defaults to True.
            max_grad_norm (float, optional): Used for gradient normalization. Defaults to 1.
            use_amp (bool, optional): Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0. Defaults to False.
            callback (Callable[[float, int, int], None], optional): Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`. Defaults to None.
            show_progress_bar (bool, optional): If True, output a tqdm progress bar. Defaults to True.
        """
        # train_dataloader.collate_fn = self.smart_batching_collate
    
        if use_amp:
            if is_torch_npu_available():
                scaler = torch.npu.amp.GradScaler()
            else:
                scaler = torch.cuda.amp.GradScaler()
        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for batch in tqdm(
                train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar
            ):
                assert len(batch) % 4 == 0
                actual_batch_size = len(batch)//4
                features, labels = self.smart_batching_collate(batch)
                if use_amp:
                    with torch.autocast(device_type=self._target_device.type):
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        
                        # t는 (B*4) x 2 크기의 텐서입니다.
                        logits = logits.view(actual_batch_size, 4, 2)  # Step 1: Reshape

                        # Step 2: idx=1 값에서 최대값의 인덱스 찾기
                        _, indices = torch.max(logits[:, :, 1], dim=1)

                        # Step 3: 최대값에 해당하는 벡터 선택
                        logits = logits[torch.arange(actual_batch_size), indices, :]
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    
                    # t는 (B*4) x 2 크기의 텐서입니다.
                    logits = logits.view(actual_batch_size, 4, 2)  # Step 1: Reshape

                    # Step 2: idx=1 값에서 최대값의 인덱스 찾기
                    _, indices = torch.max(logits[:, :, 1], dim=1)

                    # Step 3: 최대값에 해당하는 벡터 선택
                    logits = logits[torch.arange(actual_batch_size), indices, :]
                    del indices
                    
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator, output_path, save_best_model, epoch, training_steps, callback
                    )

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)


# class MoresCL(CustomCrossEncoder):
#     def fit(
#         self,
#         train_dataloader: DataLoader,
#         evaluator: SentenceEvaluator = None,
#         epochs: int = 1,
#         loss_fct=None,
#         activation_fct=nn.Identity(),
#         scheduler: str = "WarmupLinear",
#         warmup_steps: int = 10000,
#         optimizer_class: type[Optimizer] = torch.optim.AdamW,
#         optimizer_params: dict[str, object] = {"lr": 1e-5},
#         weight_decay: float = 0.01,
#         evaluation_steps: int = 0,
#         output_path: str = None,
#         save_best_model: bool = True,
#         max_grad_norm: float = 1,
#         use_amp: bool = False,
#         callback: Callable[[float, int, int], None] = None,
#         show_progress_bar: bool = True,
#     ) -> None:
#         """
#         Train the model with the given training objective
#         Each training objective is sampled in turn for one batch.
#         We sample only as many batches from each objective as there are in the smallest one
#         to make sure of equal training with each dataset.

#         Args:
#             train_dataloader (DataLoader): DataLoader with training InputExamples
#             evaluator (SentenceEvaluator, optional): An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc. Defaults to None.
#             epochs (int, optional): Number of epochs for training. Defaults to 1.
#             loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss(). Defaults to None.
#             activation_fct: Activation function applied on top of logits output of model.
#             scheduler (str, optional): Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts. Defaults to "WarmupLinear".
#             warmup_steps (int, optional): Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero. Defaults to 10000.
#             optimizer_class (Type[Optimizer], optional): Optimizer. Defaults to torch.optim.AdamW.
#             optimizer_params (Dict[str, object], optional): Optimizer parameters. Defaults to {"lr": 2e-5}.
#             weight_decay (float, optional): Weight decay for model parameters. Defaults to 0.01.
#             evaluation_steps (int, optional): If > 0, evaluate the model using evaluator after each number of training steps. Defaults to 0.
#             output_path (str, optional): Storage path for the model and evaluation files. Defaults to None.
#             save_best_model (bool, optional): If true, the best model (according to evaluator) is stored at output_path. Defaults to True.
#             max_grad_norm (float, optional): Used for gradient normalization. Defaults to 1.
#             use_amp (bool, optional): Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0. Defaults to False.
#             callback (Callable[[float, int, int], None], optional): Callback function that is invoked after each evaluation.
#                 It must accept the following three parameters in this order:
#                 `score`, `epoch`, `steps`. Defaults to None.
#             show_progress_bar (bool, optional): If True, output a tqdm progress bar. Defaults to True.
#         """
#         # train_dataloader.collate_fn = self.smart_batching_collate
    
#         if use_amp:
#             if is_torch_npu_available():
#                 scaler = torch.npu.amp.GradScaler()
#             else:
#                 scaler = torch.cuda.amp.GradScaler()
#         self.model.to(self._target_device)

#         if output_path is not None:
#             os.makedirs(output_path, exist_ok=True)

#         self.best_score = -9999999
#         num_train_steps = int(len(train_dataloader) * epochs)

#         # Prepare optimizers
#         param_optimizer = list(self.model.named_parameters())

#         no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
#         optimizer_grouped_parameters = [
#             {
#                 "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#                 "weight_decay": weight_decay,
#             },
#             {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
#         ]

#         optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

#         if isinstance(scheduler, str):
#             scheduler = SentenceTransformer._get_scheduler(
#                 optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
#             )

#         if loss_fct is None:
#             loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

#         skip_scheduler = False
#         for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
#             training_steps = 0
#             self.model.zero_grad()
#             self.model.train()

#             for batch in tqdm(
#                 train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar
#             ):
#                 features, labels = self.smart_batching_collate(batch)
#                 if use_amp:
#                     with torch.autocast(device_type=self._target_device.type):
#                         model_predictions = self.model(**features, return_dict=True)
#                         logits = activation_fct(model_predictions.logits)
#                         if self.config.num_labels == 1:
#                             logits = logits.view(-1)
#                         loss_value = loss_fct(logits, labels)

#                     scale_before_step = scaler.get_scale()
#                     scaler.scale(loss_value).backward()
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
#                     scaler.step(optimizer)
#                     scaler.update()

#                     skip_scheduler = scaler.get_scale() != scale_before_step
#                 else:
#                     model_predictions = self.model(**features, return_dict=True)
#                     logits = activation_fct(model_predictions.logits)
#                     if self.config.num_labels == 1:
#                         logits = logits.view(-1)
#                     loss_value = loss_fct(logits, labels)
#                     loss_value.backward()
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
#                     optimizer.step()

#                 optimizer.zero_grad()

#                 if not skip_scheduler:
#                     scheduler.step()

#                 training_steps += 1

#                 if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
#                     self._eval_during_training(
#                         evaluator, output_path, save_best_model, epoch, training_steps, callback
#                     )

#                     self.model.zero_grad()
#                     self.model.train()

#             if evaluator is not None:
#                 self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)


from packaging import version
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainerCallback, TrainerControl, TrainerState

from sentence_transformers.datasets.NoDuplicatesDataLoader import NoDuplicatesDataLoader
from sentence_transformers.datasets.SentenceLabelDataset import SentenceLabelDataset
from sentence_transformers.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
    SentenceTransformerTrainingArguments,
)

from sentence_transformers.util import (
    batch_to_device,
    get_device_name,
    import_from_string,
    is_sentence_transformer_model,
    load_dir_path,
    load_file_path,
    save_to_hub_args_decorator,
    truncate_embeddings,
    is_datasets_available
)

if is_datasets_available():
    from datasets import Dataset, DatasetDict

from sentence_transformers.fit_mixin import EvaluatorCallback

class SaveModelCallback(TrainerCallback):
    """A Callback to save the model to the `output_dir`.

    There are two cases:
    1. save_best_model is True and evaluator is defined:
        We save on evaluate, but only if the new model is better than the currently saved one
        according to the evaluator.
    2. If evaluator is not defined:
        We save after the model has been trained.
    """

    def __init__(self, output_dir: str, evaluator: SentenceEvaluator | None, save_best_model: bool) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.evaluator = evaluator
        self.save_best_model = save_best_model
        self.best_metric = None

    def is_better(self, new_metric: float) -> bool:
        if getattr(self.evaluator, "greater_is_better", True):
            return new_metric > self.best_metric
        return new_metric < self.best_metric

    def on_evaluate(
        self,
        args: SentenceTransformerTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any],
        model: SentenceTransformer,
        **kwargs,
    ) -> None:
        if self.evaluator is not None and self.save_best_model:
            metric_key = getattr(self.evaluator, "primary_metric", "evaluator")
            for key, value in metrics.items():
                if key.endswith(metric_key):
                    if self.best_metric is None or self.is_better(value):
                        self.best_metric = value
                        model.save(self.output_dir)

    def on_train_end(
        self,
        args: SentenceTransformerTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: SentenceTransformer,
        **kwargs,
    ) -> None:
        if self.evaluator is None:
            model.save(self.output_dir)

class SentenceTransformerCL(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        tokenizer_name = kwargs['tokenizer_name']
        del kwargs['tokenizer_name']
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def fit(
        self,
        train_objectives: Iterable[tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        steps_per_epoch=None,
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: type[Optimizer] = torch.optim.AdamW,
        optimizer_params: dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
    ) -> None:
        """
        Deprecated training method from before Sentence Transformers v3.0, it is recommended to use
        :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` instead. This method uses
        :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` behind the scenes, but does
        not provide as much flexibility as the Trainer itself.

        This training approach uses a list of DataLoaders and Loss functions to train the model. Each DataLoader
        is sampled in turn for one batch. We sample only as many batches from each DataLoader as there are in the
        smallest one to make sure of equal training with each dataset, i.e. round robin sampling.

        This method should produce equivalent results in v3.0+ as before v3.0, but if you encounter any issues
        with your existing training scripts, then you may wish to use
        :meth:`SentenceTransformer.old_fit <sentence_transformers.SentenceTransformer.old_fit>` instead.
        That uses the old training method from before v3.0.

        Args:
            train_objectives: Tuples of (DataLoader, LossFunction). Pass
                more than one for multi-task learning
            evaluator: An evaluator (sentence_transformers.evaluation)
                evaluates the model performance during training on held-
                out dev data. It is used to determine the best model
                that is saved to disc.
            epochs: Number of epochs for training
            steps_per_epoch: Number of training steps per epoch. If set
                to None (default), one epoch is equal the DataLoader
                size from train_objectives.
            scheduler: Learning rate scheduler. Available schedulers:
                constantlr, warmupconstant, warmuplinear, warmupcosine,
                warmupcosinewithhardrestarts
            warmup_steps: Behavior depends on the scheduler. For
                WarmupLinear (default), the learning rate is increased
                from o up to the maximal learning rate. After these many
                training steps, the learning rate is decreased linearly
                back to zero.
            optimizer_class: Optimizer
            optimizer_params: Optimizer parameters
            weight_decay: Weight decay for model parameters
            evaluation_steps: If > 0, evaluate the model using evaluator
                after each number of training steps
            output_path: Storage path for the model and evaluation files
            save_best_model: If true, the best model (according to
                evaluator) is stored at output_path
            max_grad_norm: Used for gradient normalization.
            use_amp: Use Automatic Mixed Precision (AMP). Only for
                Pytorch >= 1.6.0
            callback: Callback function that is invoked after each
                evaluation. It must accept the following three
                parameters in this order: `score`, `epoch`, `steps`
            show_progress_bar: If True, output a tqdm progress bar
            checkpoint_path: Folder to save checkpoints during training
            checkpoint_save_steps: Will save a checkpoint after so many
                steps
            checkpoint_save_total_limit: Total number of checkpoints to
                store
        """
        if not is_datasets_available():
            raise ImportError("Please install `datasets` to use this function: `pip install datasets`.")

        # Delayed import to counter the SentenceTransformers -> FitMixin -> SentenceTransformerTrainer -> SentenceTransformers circular import
        from sentence_transformers.trainer import SentenceTransformerTrainer

        data_loaders, loss_fns = zip(*train_objectives)

        # Clear the dataloaders from collate functions as we just want raw InputExamples
        def identity(batch):
            return batch

        # for data_loader in data_loaders:
        #     data_loader.collate_fn = identity

        batch_size = 8
        batch_sampler = BatchSamplers.BATCH_SAMPLER
        # Convert dataloaders into a DatasetDict
        # TODO: This is rather inefficient, as we load all data into memory. We might benefit from a more efficient solution
        train_dataset_dict = {}
        for loader_idx, data_loader in enumerate(data_loaders, start=1):
            if isinstance(data_loader, NoDuplicatesDataLoader):
                batch_sampler = BatchSamplers.NO_DUPLICATES
            elif hasattr(data_loader, "dataset") and isinstance(data_loader.dataset, SentenceLabelDataset):
                batch_sampler = BatchSamplers.GROUP_BY_LABEL

            batch_size = getattr(data_loader, "batch_size", batch_size)
            texts = []
            labels = []
            print("loading batch to list..")
            for batch in tqdm(data_loader):
                batch_texts, batch_labels = zip(*[(example.texts, example.label) for example in batch])
                texts += batch_texts
                labels += batch_labels
            dataset = Dataset.from_dict({f"sentence_{idx}": text for idx, text in enumerate(zip(*texts))})
            # Add label column, unless all labels are 0 (the default value for `labels` in InputExample)
            add_label_column = True
            try:
                if set(labels) == {0}:
                    add_label_column = False
            except TypeError:
                pass
            if add_label_column:
                dataset = dataset.add_column("label", labels)
            train_dataset_dict[f"_dataset_{loader_idx}"] = dataset

        train_dataset_dict = DatasetDict(train_dataset_dict)

        def _default_checkpoint_dir() -> str:
            dir_name = "checkpoints/model"
            idx = 1
            while Path(dir_name).exists() and len(list(Path(dir_name).iterdir())) != 0:
                dir_name = f"checkpoints/model_{idx}"
                idx += 1
            return dir_name

        # Convert loss_fns into a dict with `dataset_{idx}` keys
        loss_fn_dict = {f"_dataset_{idx}": loss_fn for idx, loss_fn in enumerate(loss_fns, start=1)}

        # Use steps_per_epoch to perhaps set max_steps
        max_steps = -1
        if steps_per_epoch is not None and steps_per_epoch > 0:
            if epochs == 1:
                max_steps = steps_per_epoch
            else:
                logger.warning(
                    "Setting `steps_per_epoch` alongside `epochs` > 1 no longer works. "
                    "We will train with the full datasets per epoch."
                )
                steps_per_epoch = None

        # Transformers renamed `evaluation_strategy` to `eval_strategy` in v4.41.0
        eval_strategy_key = (
            "eval_strategy"
            if version.parse(transformers.__version__) >= version.parse("4.41.0")
            else "evaluation_strategy"
        )
        args = SentenceTransformerTrainingArguments(
            output_dir=checkpoint_path or _default_checkpoint_dir(),
            batch_sampler=batch_sampler,
            multi_dataset_batch_sampler=MultiDatasetBatchSamplers.ROUND_ROBIN,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            max_steps=max_steps,
            **{
                eval_strategy_key: "steps" if evaluation_steps is not None and evaluation_steps > 0 else "no",
            },
            eval_steps=evaluation_steps,
            # load_best_model_at_end=save_best_model, # <- TODO: Look into a good solution for save_best_model
            max_grad_norm=max_grad_norm,
            fp16=use_amp,
            disable_tqdm=not show_progress_bar,
            save_strategy="steps" if checkpoint_path is not None else "no",
            save_steps=checkpoint_save_steps,
            save_total_limit=checkpoint_save_total_limit,
        )

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(train_dataset) // batch_size for train_dataset in train_dataset_dict.values()])
        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizer & scheduler
        param_optimizer = list(self.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler_obj = self._get_scheduler(
            optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
        )

        # Create callbacks
        callbacks = []
        if evaluator is not None:
            callbacks.append(EvaluatorCallback(evaluator))
            if callback is not None:
                callbacks.append(OriginalCallback(callback, evaluator))

        trainer = SentenceTransformerTrainer(
            model=self,
            args=args,
            train_dataset=train_dataset_dict,
            eval_dataset=None,
            loss=loss_fn_dict,
            evaluator=evaluator,
            optimizers=(optimizer, scheduler_obj),
            callbacks=callbacks,
        )
        # Set the trainer on the EvaluatorCallback, required for logging the metrics
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, EvaluatorCallback):
                callback.trainer = trainer

        if output_path is not None:
            trainer.add_callback(SaveModelCallback(output_path, evaluator, save_best_model))

        trainer.train()