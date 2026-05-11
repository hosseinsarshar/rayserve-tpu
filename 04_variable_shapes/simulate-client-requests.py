"""Client script to simulate concurrent requests with variable tensor shapes.

This script sends POST requests with variable length tensors to the Ray Serve endpoint
concurrently, with random delays to simulate different arrival times.
"""

import concurrent.futures
import logging
import random
import sys
import threading
import time
import requests

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

stop_event = threading.Event()
MAX_LEN = 100

def send_request(url):
    if stop_event.is_set():
        return
        
    # Simulate random arrival time by sleeping for a fraction of a second
    delay = random.uniform(0.1, 0.8)
    
    # Sleep in small increments to check stop event frequently
    slept = 0.0
    while slept < delay:
        if stop_event.is_set():
            return
        time.sleep(0.05)
        slept += 0.05
        
    # Create a random tensor of random size up to MAX_LEN
    tensor_size = random.randint(1, MAX_LEN)
    data = [float(random.randint(0, 10)) for _ in range(tensor_size)]
    
    logging.info(f"Sending tensor of size {tensor_size} after {delay:.2f}s delay")
    
    try:
        response = requests.post(url, json={"data": data})
        response.raise_for_status()
        result = response.json()['result']
        logging.info(f"Response received for size {tensor_size}. Result size: {len(result)}")
    except requests.exceptions.RequestException as e:
        if not stop_event.is_set():
            logging.error(f"Error for size {tensor_size}: {e}")

def worker(url):
    while not stop_event.is_set():
        send_request(url)
        # Wait a bit between requests for a single worker
        time.sleep(random.uniform(0.1, 0.5))

def main():
    url = "http://localhost:8000/process_variable_tensor"
    num_workers = 10
    
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
        
    logging.info(f"Starting continuous simulation with {num_workers} workers...")
    logging.info("Press Ctrl+C to stop.")
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    
    # Start worker threads
    for i in range(num_workers):
        executor.submit(worker, url)
        
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("\nStopping simulation...")
        stop_event.set()
        executor.shutdown(wait=True)
        logging.info("Simulation stopped.")

if __name__ == "__main__":
    main()
