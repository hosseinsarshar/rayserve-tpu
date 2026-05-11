"""Client script to simulate concurrent requests with random delays.

This script sends multiple requests to the Ray Serve endpoint concurrently,
with random delays to simulate different arrival times. This showcases how
dynamic batching groups them on the server side.
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

# Event to signal workers to stop
stop_event = threading.Event()

def send_request(url, value):
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
        
    logging.info(f"Sending request for value {value} after {delay:.2f}s delay")
    
    try:
        response = requests.get(url, params={"value": value})
        response.raise_for_status()
        logging.info(f"Response for value {value}: {response.json()['result']}")
    except requests.exceptions.RequestException as e:
        if not stop_event.is_set():
            logging.error(f"Error for value {value}: {e}")

def worker(url):
    while not stop_event.is_set():
        value = float(random.randint(0, 100))
        send_request(url, value)
        # Wait a bit between requests for a single worker
        time.sleep(random.uniform(0.1, 0.5))

def main():
    url = "http://localhost:8000/add_one"
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
