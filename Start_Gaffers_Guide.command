#!/usr/bin/env bash

# Change to the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================="
echo " Starting Gaffer's Guide Development..."
echo "========================================="

# Trap SIGINT and SIGTERM to clean up background processes
trap "echo 'Shutting down...'; kill 0; exit" SIGINT SIGTERM EXIT

# Start the Backend in the background
echo "-> Starting FastAPI Backend..."
cd "$DIR/backend"
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
fi
export PYTHONPATH="$DIR/backend:$DIR/src"
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &

# Wait a few seconds for the backend to start
sleep 3

# Start the Frontend
echo "-> Starting Electron Frontend..."
cd "$DIR/frontend_final"
npm run electron:dev
