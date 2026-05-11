"""Client script to test the tensor serving endpoint.
"""

import random
import sys
import requests

def test_tensor_serve():
    url = "http://localhost:8000/process_tensor"
    TENSOR_SIZE = 10
    
    # Create a random tensor of size 10
    data = [float(random.randint(0, 10)) for _ in range(TENSOR_SIZE)]
    
    print(f"Sending tensor: {data}")
    try:
        response = requests.post(url, json={"data": data})
        response.raise_for_status()
        print("Response received:")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    test_tensor_serve()
