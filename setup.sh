#!/bin/bash

echo "🚀 Setting up Instagram Clothing Bot..."

# Create necessary directories
mkdir -p downloads
mkdir -p inventory_images

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your Instagram credentials!"
fi

# Create inventory images directory
echo "📸 Creating inventory images directory..."
mkdir -p inventory_images

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your Instagram credentials"
echo "2. Add product images to inventory_images/ directory"
echo "3. Run: python ai_service.py (in one terminal)"
echo "4. Run: npm start (in another terminal)"
echo ""
echo "⚠️  Important: Use a test Instagram account, not your main account!"