import asyncio
import httpx
import time

URL = "http://127.0.0.1:8001/api/v1/analyze"
FILE_PATH = "test_video.mp4"
CONCURRENT_REQUESTS = 10

async def send_request(client, req_id):
    start = time.perf_counter()
    with open(FILE_PATH, "rb") as f:
        files = {"file": ("test_video.mp4", f, "video/mp4")}
        response = await client.post(URL, files=files)
        latency = (time.perf_counter() - start) * 1000
        print(f"Req {req_id}: Status {response.status_code} | Total HTTP Time: {latency:.2f}ms | Server Engine: {response.json().get('execution_engine')}")

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [send_request(client, i) for i in range(CONCURRENT_REQUESTS)]
        start_total = time.perf_counter()
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_total
        print(f"\n--- Load Test Complete ---")
        print(f"Processed {CONCURRENT_REQUESTS} concurrent requests in {total_time:.2f} seconds.")
        print(f"Throughput: {CONCURRENT_REQUESTS / total_time:.2f} req/sec")

if __name__ == "__main__":
    asyncio.run(main())
