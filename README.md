# LLM Fine-Tuner API

A comprehensive FastAPI-based REST API service for fine-tuning Large Language Models (LLMs).

## Features

- 🤗 **Multiple Model Support**: Fine-tune models from Hugging Face Hub including Llama 3, Mistral, Phi-3, Gemma, BLOOM, Falcon, and more
- ⚡ **Efficient Fine-tuning Methods**: Support for LoRA, QLoRA (4-bit quantization), and Full fine-tuning
- 📦 **Flexible Dataset Handling**: Accept JSON, JSONL, CSV, Parquet files or load directly from Hugging Face datasets
- 🔄 **Async Job Processing**: Background training with real-time progress tracking
- 🧪 **Model Testing**: Test your fine-tuned models with custom prompts
- 📤 **Easy Deployment**: Download trained models or push directly to Hugging Face Hub
- 🔒 **Security**: API key authentication for all sensitive endpoints
- 📊 **Automatic Documentation**: Swagger UI and ReDoc available

## Requirements

- Python 3.11+
- CUDA-compatible GPU (recommended, 16GB+ VRAM for 7B models)
- 32GB+ RAM

## Installation

### 1. Clone or create the project

```bash
cd llm-finetuner-api
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install PyTorch (with CUDA support)

Choose the appropriate command for your CUDA version:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended for training)
pip install torch
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

Important configurations:
- `API_KEYS`: Set secure API keys for authentication
- `HF_TOKEN`: Your Hugging Face token for private models

### 6. Run the server

```bash
# Development
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

## API Documentation

Once running, access the documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Quick Start Guide

### 1. Get your API key ready

Add the `X-API-Key` header to all requests:
```bash
-H "X-API-Key: your-api-key"
```

### 2. Create a fine-tuning job

Using a Hugging Face dataset:
```bash
curl -X POST "http://localhost:8000/api/v1/jobs/create-json" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "dataset_source": "tatsu-lab/alpaca",
    "method": "lora",
    "hyperparameters": {
      "num_train_epochs": 1,
      "per_device_train_batch_size": 4,
      "learning_rate": 0.0002,
      "lora_rank": 16
    },
    "description": "Fine-tuning Phi-3 on Alpaca"
  }'
```

With file upload:
```bash
curl -X POST "http://localhost:8000/api/v1/jobs/create" \
  -H "X-API-Key: your-api-key" \
  -F "model_name=microsoft/Phi-3-mini-4k-instruct" \
  -F "dataset_source=upload" \
  -F "dataset_file=@my_dataset.jsonl" \
  -F "method=lora" \
  -F 'hyperparameters_json={"num_train_epochs": 3}'
```

### 3. Monitor job progress

```bash
curl -X GET "http://localhost:8000/api/v1/jobs/{job_id}" \
  -H "X-API-Key: your-api-key"
```

### 4. Test the trained model

```bash
curl -X POST "http://localhost:8000/api/v1/jobs/{job_id}/test" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "prompt": "What is machine learning?",
    "max_new_tokens": 256,
    "temperature": 0.7
  }'
```

### 5. Download the model

```bash
# Download LoRA adapter
curl -X GET "http://localhost:8000/api/v1/jobs/{job_id}/download" \
  -H "X-API-Key: your-api-key" \
  -o model.zip

# Download merged model (LoRA + base model)
curl -X GET "http://localhost:8000/api/v1/jobs/{job_id}/download?merge=true" \
  -H "X-API-Key: your-api-key" \
  -o model_merged.zip
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/jobs` | List all jobs |
| POST | `/api/v1/jobs/create` | Create job (with file upload) |
| POST | `/api/v1/jobs/create-json` | Create job (JSON body) |
| GET | `/api/v1/jobs/{job_id}` | Get job details |
| POST | `/api/v1/jobs/{job_id}/cancel` | Cancel running job |
| GET | `/api/v1/jobs/{job_id}/download` | Download trained model |
| POST | `/api/v1/jobs/{job_id}/test` | Test trained model |
| DELETE | `/api/v1/jobs/{job_id}` | Delete job |
| GET | `/api/v1/models/supported` | List supported models |
| GET | `/api/v1/models/device-info` | Get device information |

## Dataset Formats

### Instruction Format (Recommended)
```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  }
]
```

### Text Format
```json
[
  {"text": "### Instruction:\nWhat is AI?\n\n### Response:\nAI stands for..."}
]
```

### ShareGPT/Conversation Format
```json
[
  {
    "conversations": [
      {"from": "human", "value": "Hello!"},
      {"from": "gpt", "value": "Hi there!"}
    ]
  }
]
```

### Messages Format (OpenAI style)
```json
[
  {
    "messages": [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi there!"}
    ]
  }
]
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 2e-4 | Learning rate for training |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `num_train_epochs` | 3 | Number of training epochs |
| `max_seq_length` | 2048 | Maximum sequence length |
| `lora_rank` | 16 | LoRA rank (r) |
| `lora_alpha` | 32 | LoRA alpha scaling |
| `lora_dropout` | 0.05 | LoRA dropout |
| `warmup_steps` | 100 | Warmup steps |
| `weight_decay` | 0.01 | Weight decay |
| `gradient_accumulation_steps` | 4 | Gradient accumulation |
| `fp16` | false | Use FP16 precision |
| `bf16` | false | Use BF16 precision |

## Fine-tuning Methods

### LoRA (Low-Rank Adaptation) - Recommended
- Memory efficient (~50% of full fine-tuning)
- Fast training
- Easy to merge with base model
- Best for most use cases

### QLoRA (Quantized LoRA)
- 4-bit quantization
- ~75% memory reduction
- Slightly slower training
- Best for limited GPU memory

### Full Fine-tuning
- Updates all model weights
- Requires significant GPU memory
- Best results but resource intensive
- Recommended only with multiple GPUs

## GPU Memory Requirements

| Model Size | LoRA | QLoRA | Full |
|------------|------|-------|------|
| 3B | ~8GB | ~4GB | ~24GB |
| 7B | ~16GB | ~8GB | ~56GB |
| 13B | ~32GB | ~16GB | ~104GB |
| 70B | ~160GB | ~48GB | ~560GB |

## Project Structure

```
llm-finetuner-api/
├── main.py                  # FastAPI application
├── api/
│   └── v1/
│       ├── router.py        # API router
│       └── endpoints.py     # API endpoints
├── core/
│   ├── config.py            # Configuration settings
│   ├── security.py          # API key validation
│   └── dependencies.py      # Database dependencies
├── models/
│   └── job.py               # Pydantic models
├── services/
│   ├── finetune_service.py  # Fine-tuning logic
│   ├── dataset_service.py   # Dataset processing
│   └── storage_service.py   # Storage management
├── utils/
│   ├── logging.py           # Logging utilities
│   └── helpers.py           # Helper functions
├── data/
│   ├── uploads/             # Uploaded datasets
│   ├── jobs/                # Job metadata/logs
│   └── logs/                # Application logs
├── trained_models/          # Fine-tuned models
├── requirements.txt
├── .env.example
└── README.md
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use QLoRA instead of LoRA
- Reduce `max_seq_length`

### Model Not Found
- Check if `HF_TOKEN` is set for gated models
- Accept model license on Hugging Face website
- Verify model name is correct

### Slow Training
- Enable `fp16` or `bf16` if GPU supports it
- Increase batch size if memory allows
- Use Flash Attention if available

## License

This project is for educational purposes.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
