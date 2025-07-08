#!/bin/bash
set -e

echo "🧪 Checking virtual environment..."
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "❌ No virtual environment detected."
    echo "💡 Please activate one before running this script:"
    echo ""
    echo "   python3 -m venv .venv"
    echo "   source .venv/bin/activate"
    echo ""
    echo "Then re-run ./setup.sh"
    exit 1
fi

echo "📁 Setting up dependencies..."
cd ..

# Clone and install fifo-dev-common
if [ ! -d "fifo-dev-common" ]; then
    echo "🔁 Cloning fifo-dev-common..."
    git clone https://github.com/gh9869827/fifo-dev-common.git
fi
echo "📦 Installing fifo-dev-common..."
python3 -m pip install -e fifo-dev-common

# Clone and install fifo-tool-airlock-model-env
if [ ! -d "fifo-tool-airlock-model-env" ]; then
    echo "🔁 Cloning fifo-tool-airlock-model-env..."
    git clone https://github.com/gh9869827/fifo-tool-airlock-model-env.git
fi
echo "📦 Installing airlock SDK..."
python3 -m pip install -e fifo-tool-airlock-model-env[sdk]
echo "📦 Installing airlock bridge..."
python3 -m pip install -e fifo-tool-airlock-model-env[bridge]

# Go back to DSL repo and install it
cd fifo-dev-dsl
echo "📦 Installing DSL module..."
python3 -m pip install -e .

echo "✅ Setup complete!"
