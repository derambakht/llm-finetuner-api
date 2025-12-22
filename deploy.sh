#!/bin/bash

# ============================================
# LLM Fine-Tuner API - Server Deployment Script
# ============================================

set -e

echo "🚀 Starting LLM Fine-Tuner API Deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Installing..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker installed successfully"
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Installing..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed successfully"
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected"
    GPU_AVAILABLE=true
    
    # Check if nvidia-docker is installed
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        echo "📦 Installing NVIDIA Container Toolkit..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
        echo "✅ NVIDIA Container Toolkit installed"
    fi
else
    echo "💻 No NVIDIA GPU detected, will run on CPU"
    GPU_AVAILABLE=false
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.docker.example .env
    echo "⚠️  Please edit .env file and set your HF_TOKEN and API_KEYS"
    echo "   Run: nano .env"
fi

# Create necessary directories
mkdir -p data/uploads data/jobs data/logs trained_models

# Build and run
echo "🔨 Building Docker image..."
if [ "$GPU_AVAILABLE" = true ]; then
    echo "🎮 Starting with GPU support..."
    docker-compose --profile gpu up -d --build
else
    echo "💻 Starting with CPU only..."
    docker-compose --profile cpu up -d --build
fi

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📡 API is running at: http://$(hostname -I | awk '{print $1}'):8000"
echo "📖 API Docs: http://$(hostname -I | awk '{print $1}'):8000/docs"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop: docker-compose down"
echo "  - Restart: docker-compose restart"
