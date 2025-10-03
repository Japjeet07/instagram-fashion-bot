#!/bin/bash

echo "🚀 Starting Instagram Clothing Bot..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please run ./setup.sh first"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the bot
echo "🤖 Starting bot..."
echo "📨 Send DMs to your bot account to test!"
echo ""
echo "Press Ctrl+C to stop the bot"

node bot.js