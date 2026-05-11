import requests
import sys

def test_serve(value=5.0):
    url = "http://localhost:8000/add_one"
    params = {"value": value}
    
    try:
        print(f"Sending request to {url} with value={value}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        print("Response received:")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    val = 5.0
    if len(sys.argv) > 1:
        val = float(sys.argv[1])
    test_serve(val)
