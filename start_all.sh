#!/bin/bash

echo "🚀 Starting Instagram Clothing Bot System..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found! Please copy env.example to .env and add your credentials."
    exit 1
fi

# Start AI Service (Python)
echo "🤖 Starting AI Service..."
python3 ai_service.py &
AI_PID=$!

# Wait for AI service to start
sleep 5

# Start Frontend
echo "⚛️ Starting Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 10

# Start Instagram Bot
echo "📱 Starting Instagram Bot..."
npm start &
BOT_PID=$!

echo "✅ All services started!"
echo "🤖 AI Service: http://localhost:5000"
echo "⚛️ Frontend: http://localhost:8080"
echo "📱 Instagram Bot: Running"
echo ""
echo "📋 Process IDs:"
echo "   AI Service: $AI_PID"
echo "   Frontend: $FRONTEND_PID"
echo "   Instagram Bot: $BOT_PID"
echo ""
echo "🛑 To stop all services: kill $AI_PID $FRONTEND_PID $BOT_PID"

# Keep script running
wait
