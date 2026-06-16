import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws/jobs/0812efcb80324fc3bab637e03e636b81"
    try:
        async with websockets.connect(uri) as websocket:
            message = await websocket.recv()
            print(message)
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test_ws())
