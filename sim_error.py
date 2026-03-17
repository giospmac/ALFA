import urllib.request
try:
    with urllib.request.urlopen('http://localhost:8501/healthz') as response:
        print(response.read().decode())
except Exception as e:
    print(e)
