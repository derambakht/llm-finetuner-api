"""
Fine-tuning service implementing the core training logic.
ML libraries are lazily imported to allow API startup without GPU.
"""
import os
import gc
import threading
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

# ML libraries are imported lazily in _import_ml_libraries()

from core.config import settings
from models.job import JobStatus, FineTuneMethod, Hyperparameters
from services.storage_service import storage_service
from services.dataset_service import dataset_service
from utils.logging import get_logger, JobLogger
from utils.helpers import (
    get_device_info,
    get_default_lora_target_modules,
    estimate_model_memory_gb,
)

logger = get_logger(__name__)


# Track running jobs for cancellation
running_jobs: Dict[str, threading.Event] = {}

# ML libraries cache
_ml_libs_cache: Optional[Dict[str, Any]] = None


def _import_ml_libraries() -> Dict[str, Any]:
    """Lazily import ML libraries only when needed for training."""
    global _ml_libs_cache
    if _ml_libs_cache is not None:
        return _ml_libs_cache
    
    logger.info("Loading ML libraries (torch, transformers, peft, trl)...")
    
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        TrainerCallback,
        TrainerState,
        TrainerControl,
        BitsAndBytesConfig,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
        TaskType,
    )
    from trl import SFTTrainer, SFTConfig
    from huggingface_hub import HfApi
    
    _ml_libs_cache = {
        'torch': torch,
        'AutoModelForCausalLM': AutoModelForCausalLM,
        'AutoTokenizer': AutoTokenizer,
        'TrainingArguments': TrainingArguments,
        'TrainerCallback': TrainerCallback,
        'TrainerState': TrainerState,
        'TrainerControl': TrainerControl,
        'BitsAndBytesConfig': BitsAndBytesConfig,
        'LoraConfig': LoraConfig,
        'get_peft_model': get_peft_model,
        'prepare_model_for_kbit_training': prepare_model_for_kbit_training,
        'PeftModel': PeftModel,
        'TaskType': TaskType,
        'SFTTrainer': SFTTrainer,
        'SFTConfig': SFTConfig,
        'HfApi': HfApi,
    }
    
    logger.info("ML libraries loaded successfully.")
    return _ml_libs_cache


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning job."""
    
    job_id: str
    model_name: str
    dataset_source: str
    dataset_path: Optional[str]
    method: FineTuneMethod
    hyperparameters: Hyperparameters
    hf_token: Optional[str] = None
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    output_dir: str = field(default="")
    
    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = storage_service.get_model_output_path(self.job_id)


def _create_progress_callback_class():
    """Create ProgressCallback class that inherits from TrainerCallback."""
    ml = _import_ml_libraries()
    TrainerCallback = ml['TrainerCallback']
    
    class ProgressCallback(TrainerCallback):
        """Callback to track training progress and handle cancellation."""
        
        def __init__(
            self, 
            job_id: str, 
            job_logger: JobLogger,
            cancel_event: threading.Event,
            progress_callback: Optional[Callable[[float], None]] = None
        ):
            super().__init__()
            self.job_id = job_id
            self.job_logger = job_logger
            self.cancel_event = cancel_event
            self.progress_callback = progress_callback
            self.total_steps = 0
            self.current_step = 0
        
        def on_train_begin(self, args, state, control, **kwargs):
            self.total_steps = state.max_steps
            self.job_logger.info(f"Training started. Total steps: {self.total_steps}")
            storage_service.add_job_log(self.job_id, "INFO", f"Training started. Total steps: {self.total_steps}")
        
        def on_step_end(self, args, state, control, **kwargs):
            self.current_step = state.global_step
            
            # Check for cancellation
            if self.cancel_event.is_set():
                self.job_logger.info("Cancellation requested. Stopping training...")
                control.should_training_stop = True
                return
            
            # Update progress
            if self.total_steps > 0:
                progress = (self.current_step / self.total_steps) * 100
                storage_service.update_job_progress(self.job_id, progress)
                
                if self.progress_callback:
                    self.progress_callback(progress)
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                log_msg = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                    for k, v in logs.items() if k != "total_flos"])
                self.job_logger.info(f"Step {state.global_step}: {log_msg}")
                storage_service.add_job_log(self.job_id, "INFO", f"Step {state.global_step}: {log_msg}")
        
        def on_train_end(self, args, state, control, **kwargs):
            self.job_logger.info("Training completed.")
            storage_service.add_job_log(self.job_id, "INFO", "Training completed.")
    
    return ProgressCallback


class FineTuneService:
    """Service for managing fine-tuning operations."""
    
    def __init__(self):
        self.device_info = get_device_info()
        logger.info(f"Device info: {self.device_info}")
    
    def start_training(self, config: FineTuneConfig) -> None:
        """
        Start fine-tuning in a background thread.
        
        Args:
            config: Fine-tuning configuration
        """
        # Create cancellation event
        cancel_event = threading.Event()
        running_jobs[config.job_id] = cancel_event
        
        # Start training thread
        thread = threading.Thread(
            target=self._run_training,
            args=(config, cancel_event),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started training thread for job {config.job_id}")
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Request cancellation of a running job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancellation was requested
        """
        if job_id in running_jobs:
            running_jobs[job_id].set()
            logger.info(f"Cancellation requested for job {job_id}")
            return True
        return False
    
    def _run_training(self, config: FineTuneConfig, cancel_event: threading.Event) -> None:
        """
        Execute the training process.
        
        Args:
            config: Fine-tuning configuration
            cancel_event: Event for cancellation
        """
        job_logger = JobLogger(config.job_id)
        ml = None  # Will be loaded when needed
        
        try:
            # Update status to running
            storage_service.update_job_status(config.job_id, JobStatus.RUNNING)
            job_logger.info(f"Starting fine-tuning job: {config.job_id}")
            job_logger.info(f"Model: {config.model_name}")
            job_logger.info(f"Method: {config.method.value}")
            job_logger.info(f"Device: {self.device_info['device']}")
            
            storage_service.add_job_log(config.job_id, "INFO", f"Loading model: {config.model_name}")
            
            # Prepare dataset
            job_logger.info("Preparing dataset...")
            train_dataset, eval_dataset = dataset_service.prepare_dataset(
                source=config.dataset_source,
                file_path=config.dataset_path,
                token=config.hf_token,
            )
            job_logger.info(f"Training samples: {len(train_dataset)}")
            if eval_dataset:
                job_logger.info(f"Evaluation samples: {len(eval_dataset)}")
            
            storage_service.add_job_log(config.job_id, "INFO", f"Dataset prepared: {len(train_dataset)} samples")
            
            # Check for cancellation
            if cancel_event.is_set():
                self._handle_cancellation(config.job_id, job_logger)
                return
            
            # Load ML libraries now (lazy loading)
            ml = _import_ml_libraries()
            
            # Load tokenizer
            job_logger.info("Loading tokenizer...")
            tokenizer = self._load_tokenizer(config.model_name, config.hf_token, ml)
            
            # Load model
            job_logger.info("Loading model...")
            model = self._load_model(config, ml)
            
            storage_service.add_job_log(config.job_id, "INFO", "Model loaded successfully")
            
            # Check for cancellation
            if cancel_event.is_set():
                self._handle_cancellation(config.job_id, job_logger)
                return
            
            # Configure and start training
            job_logger.info("Configuring trainer...")
            trainer = self._create_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                config=config,
                job_logger=job_logger,
                cancel_event=cancel_event,
                ml=ml,
            )
            
            # Train
            job_logger.info("Starting training...")
            storage_service.add_job_log(config.job_id, "INFO", "Training started")
            
            trainer.train()
            
            # Check if cancelled
            if cancel_event.is_set():
                self._handle_cancellation(config.job_id, job_logger)
                return
            
            # Save model
            job_logger.info("Saving model...")
            self._save_model(trainer, config, job_logger)
            
            # Push to Hub if requested
            if config.push_to_hub and config.hub_model_id and config.hf_token:
                job_logger.info("Pushing to Hugging Face Hub...")
                self._push_to_hub(config, job_logger, ml)
            
            # Mark as completed
            storage_service.update_job_status(
                config.job_id,
                JobStatus.COMPLETED,
                progress=100.0,
                output_path=config.output_dir
            )
            storage_service.add_job_log(config.job_id, "INFO", "Training completed successfully")
            job_logger.info("Training completed successfully!")
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            job_logger.error(error_msg)
            storage_service.update_job_status(
                config.job_id,
                JobStatus.FAILED,
                error_message=str(e)
            )
            storage_service.add_job_log(config.job_id, "ERROR", error_msg)
            logger.error(f"Job {config.job_id} failed: {e}")
        
        finally:
            # Cleanup
            if config.job_id in running_jobs:
                del running_jobs[config.job_id]
            
            # Free memory
            gc.collect()
            if ml is not None:
                torch = ml['torch']
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def _load_tokenizer(self, model_name: str, token: Optional[str], ml: Dict[str, Any]):
        """Load and configure tokenizer."""
        AutoTokenizer = ml['AutoTokenizer']
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token or settings.HF_TOKEN,
            trust_remote_code=True,
            cache_dir=settings.HF_CACHE_DIR,
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        tokenizer.padding_side = "right"
        
        return tokenizer
    
    def _load_model(self, config: FineTuneConfig, ml: Dict[str, Any]):
        """Load model based on fine-tuning method."""
        torch = ml['torch']
        AutoModelForCausalLM = ml['AutoModelForCausalLM']
        BitsAndBytesConfig = ml['BitsAndBytesConfig']
        LoraConfig = ml['LoraConfig']
        TaskType = ml['TaskType']
        get_peft_model = ml['get_peft_model']
        prepare_model_for_kbit_training = ml['prepare_model_for_kbit_training']
        
        device_map = "auto" if self.device_info["cuda_available"] else None
        
        # Determine dtype
        if config.hyperparameters.bf16 and self.device_info.get("bf16_supported"):
            torch_dtype = torch.bfloat16
        elif config.hyperparameters.fp16 and self.device_info.get("fp16_supported"):
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        if config.method == FineTuneMethod.QLORA:
            # QLoRA: 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                token=config.hf_token or settings.HF_TOKEN,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                cache_dir=settings.HF_CACHE_DIR,
            )
            model = prepare_model_for_kbit_training(model)
            
        elif config.method == FineTuneMethod.LORA:
            # LoRA: Load in half precision
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                token=config.hf_token or settings.HF_TOKEN,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                cache_dir=settings.HF_CACHE_DIR,
            )
            
        else:
            # Full fine-tuning
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                token=config.hf_token or settings.HF_TOKEN,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                cache_dir=settings.HF_CACHE_DIR,
            )
        
        # Enable gradient checkpointing for memory efficiency
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        
        # Apply LoRA if needed
        if config.method in [FineTuneMethod.LORA, FineTuneMethod.QLORA]:
            target_modules = config.hyperparameters.lora_target_modules or \
                           get_default_lora_target_modules(config.model_name)
            
            lora_config = LoraConfig(
                r=config.hyperparameters.lora_rank,
                lora_alpha=config.hyperparameters.lora_alpha,
                lora_dropout=config.hyperparameters.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model
    
    def _create_trainer(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        config: FineTuneConfig,
        job_logger: JobLogger,
        cancel_event: threading.Event,
        ml: Dict[str, Any],
    ):
        """Create and configure the SFT trainer."""
        SFTConfig = ml['SFTConfig']
        SFTTrainer = ml['SFTTrainer']
        
        hp = config.hyperparameters
        
        # Training arguments
        training_args = SFTConfig(
            output_dir=config.output_dir,
            num_train_epochs=hp.num_train_epochs,
            per_device_train_batch_size=hp.per_device_train_batch_size,
            gradient_accumulation_steps=hp.gradient_accumulation_steps,
            learning_rate=hp.learning_rate,
            weight_decay=hp.weight_decay,
            warmup_steps=hp.warmup_steps,
            max_seq_length=hp.max_seq_length,
            fp16=hp.fp16 and self.device_info.get("fp16_supported", False),
            bf16=hp.bf16 and self.device_info.get("bf16_supported", False),
            logging_steps=hp.logging_steps,
            save_steps=hp.save_steps,
            save_total_limit=2,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=hp.eval_steps if eval_dataset else None,
            max_steps=hp.max_steps if hp.max_steps > 0 else -1,
            report_to="none",
            optim="adamw_torch",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_text_field="text",
            packing=False,
        )
        
        # Progress callback - create dynamically to inherit from TrainerCallback
        ProgressCallback = _create_progress_callback_class()
        progress_callback = ProgressCallback(
            job_id=config.job_id,
            job_logger=job_logger,
            cancel_event=cancel_event,
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[progress_callback],
        )
        
        return trainer
    
    def _save_model(self, trainer, config: FineTuneConfig, job_logger: JobLogger) -> None:
        """Save the trained model."""
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Save the adapter (LoRA/QLoRA) or full model
        trainer.save_model(config.output_dir)
        
        # Save tokenizer
        trainer.processing_class.save_pretrained(config.output_dir)
        
        # Save training config
        config_path = os.path.join(config.output_dir, "training_config.json")
        import json
        with open(config_path, "w") as f:
            json.dump({
                "model_name": config.model_name,
                "method": config.method.value,
                "hyperparameters": config.hyperparameters.model_dump(),
            }, f, indent=2)
        
        job_logger.info(f"Model saved to {config.output_dir}")
    
    def _push_to_hub(self, config: FineTuneConfig, job_logger: JobLogger, ml: Dict[str, Any]) -> None:
        """Push model to Hugging Face Hub."""
        HfApi = ml['HfApi']
        
        try:
            api = HfApi(token=config.hf_token)
            api.upload_folder(
                folder_path=config.output_dir,
                repo_id=config.hub_model_id,
                repo_type="model",
                commit_message=f"Upload fine-tuned model from job {config.job_id}",
            )
            job_logger.info(f"Model pushed to Hub: {config.hub_model_id}")
            storage_service.add_job_log(config.job_id, "INFO", f"Model pushed to Hub: {config.hub_model_id}")
        except Exception as e:
            job_logger.error(f"Failed to push to Hub: {e}")
            storage_service.add_job_log(config.job_id, "WARNING", f"Failed to push to Hub: {e}")
    
    def _handle_cancellation(self, job_id: str, job_logger: JobLogger) -> None:
        """Handle job cancellation."""
        job_logger.info("Job cancelled by user.")
        storage_service.update_job_status(job_id, JobStatus.CANCELLED)
        storage_service.add_job_log(job_id, "INFO", "Job cancelled by user")
    
    def test_model(
        self,
        job_id: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> Dict[str, Any]:
        """
        Test a fine-tuned model with a prompt.
        
        Args:
            job_id: Job identifier
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            do_sample: Whether to use sampling
            
        Returns:
            Generated output
        """
        ml = _import_ml_libraries()
        torch = ml['torch']
        AutoTokenizer = ml['AutoTokenizer']
        AutoModelForCausalLM = ml['AutoModelForCausalLM']
        PeftModel = ml['PeftModel']
        
        model_path = storage_service.get_model_output_path(job_id)
        
        if not storage_service.model_exists(job_id):
            raise ValueError(f"Model not found for job {job_id}")
        
        # Load job info to get base model
        job = storage_service.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        base_model_name = job["model_name"]
        method = job["method"]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        # Load model
        if method in ["lora", "qlora"]:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                cache_dir=settings.HF_CACHE_DIR,
            )
            # Load adapter
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
        
        model.eval()
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 50,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "output": generated_text,
            "prompt": prompt,
            "tokens_generated": outputs[0].shape[0] - inputs["input_ids"].shape[1],
        }
    
    def merge_adapter(self, job_id: str) -> str:
        """
        Merge LoRA adapter with base model.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Path to merged model
        """
        ml = _import_ml_libraries()
        torch = ml['torch']
        AutoTokenizer = ml['AutoTokenizer']
        AutoModelForCausalLM = ml['AutoModelForCausalLM']
        PeftModel = ml['PeftModel']
        
        job = storage_service.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        if job["method"] not in ["lora", "qlora"]:
            raise ValueError("Merge only available for LoRA/QLoRA models")
        
        adapter_path = storage_service.get_model_output_path(job_id)
        merged_path = os.path.join(storage_service.models_dir, f"{job_id}_merged")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            job["model_name"],
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            cache_dir=settings.HF_CACHE_DIR,
        )
        
        # Load and merge adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = model.merge_and_unload()
        
        # Save merged model
        os.makedirs(merged_path, exist_ok=True)
        merged_model.save_pretrained(merged_path)
        
        # Copy tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        tokenizer.save_pretrained(merged_path)
        
        # Cleanup
        del model, merged_model, base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Merged model saved to {merged_path}")
        return merged_path


# Singleton instance
finetune_service = FineTuneService()
