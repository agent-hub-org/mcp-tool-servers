"""Launch all MCP tool servers for local development."""

import subprocess
import sys
import signal
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SERVERS = [
    ("web-search", "servers/web_search.py", 8010),
    ("finance-data", "servers/finance_data.py", 8011),
    ("vector-db", "servers/vector_db_server.py", 8012),
]


def main():
    processes = []

    for name, script, port in SERVERS:
        print(f"Starting {name} on port {port}...")
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / script)],
            cwd=str(ROOT),
            env={**os.environ},
        )
        processes.append((name, proc))

    print(f"\nAll {len(processes)} MCP servers started. Press Ctrl+C to stop.\n")

    def shutdown(signum, frame):
        print("\nShutting down servers...")
        for name, proc in processes:
            proc.terminate()
        for name, proc in processes:
            proc.wait(timeout=5)
            print(f"  {name} stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait for any process to exit
    while True:
        for name, proc in processes:
            ret = proc.poll()
            if ret is not None:
                print(f"ERROR: {name} exited unexpectedly with code {ret}")
                # Shut down remaining servers and exit
                for other_name, other_proc in processes:
                    if other_proc.poll() is None:
                        other_proc.terminate()
                sys.exit(1)
        try:
            signal.pause()
        except AttributeError:
            # Windows fallback
            import time
            time.sleep(1)


if __name__ == "__main__":
    main()
