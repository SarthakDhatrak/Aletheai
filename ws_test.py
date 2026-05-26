import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://localhost:8000/ws") as websocket:
        for _ in range(5):
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Prediction: {data['prediction']}, SimState: {data['sim_state']}, is_ood: {data['is_ood']}, ood_score: {data['ood_score']}")

asyncio.run(test())
