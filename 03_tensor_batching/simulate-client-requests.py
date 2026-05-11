"""Client script to simulate concurrent tensor requests with random delays.

This script sends multiple POST requests with tensor data to the Ray Serve endpoint
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
TENSOR_SIZE = 10

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
        
    # Create a random tensor of size 10
    data = [float(random.randint(0, 10)) for _ in range(TENSOR_SIZE)]
    
    logging.info(f"Sending tensor after {delay:.2f}s delay")
    
    try:
        response = requests.post(url, json={"data": data})
        response.raise_for_status()
        # Just log the first few elements of the result to avoid clutter
        result = response.json()['result']
        logging.info(f"Response received. First 3 elements: {result[:3]}")
    except requests.exceptions.RequestException as e:
        if not stop_event.is_set():
            logging.error(f"Error: {e}")

def worker(url):
    while not stop_event.is_set():
        send_request(url)
        # Wait a bit between requests for a single worker
        time.sleep(random.uniform(0.1, 0.5))

def main():
    url = "http://localhost:8000/process_tensor"
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
