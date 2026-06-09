import subprocess
import sys
import os

# Set buffer to unbuffered
os.environ["PYTHONUNBUFFERED"] = "1"

def run():
    print("Starting fetcher with verbose real-time logging...")
    process = subprocess.Popen(
        [sys.executable, "-m", "src.ingestion.fetcher"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    try:
        for line in iter(process.stdout.readline, ''):
            print(f"[FETCHER] {line.strip()}")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("Stopping fetcher...")
        process.kill()
        
    process.wait()
    print(f"Fetcher finished with exit code {process.returncode}")

if __name__ == '__main__':
    run()
