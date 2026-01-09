#!/usr/bin/env python3
"""Wait for Docker services to be ready before proceeding."""

import socket
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def check_neo4j(host: str = "localhost", port: int = 7474, timeout: float = 5.0) -> bool:
    """Check if Neo4j browser is responding."""
    try:
        response = urlopen(f"http://{host}:{port}", timeout=timeout)
        return response.status == 200
    except (URLError, TimeoutError):
        return False


def check_weaviate(host: str = "localhost", port: int = 8080, timeout: float = 5.0) -> bool:
    """Check if Weaviate is ready."""
    try:
        response = urlopen(f"http://{host}:{port}/v1/.well-known/ready", timeout=timeout)
        return response.status == 200
    except (URLError, TimeoutError):
        return False


def check_redis(host: str = "localhost", port: int = 6379, timeout: float = 5.0) -> bool:
    """Check if Redis is responding to PING."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.sendall(b"PING\r\n")
        response = sock.recv(1024)
        sock.close()
        return b"+PONG" in response
    except (OSError, TimeoutError):
        return False


def wait_for_service(
    name: str,
    check_fn: callable,
    max_attempts: int = 30,
    interval: float = 2.0,
) -> bool:
    """Wait for a service to become ready."""
    print(f"Waiting for {name}...", end="", flush=True)

    for attempt in range(max_attempts):
        if check_fn():
            print(f" ready! ({attempt + 1} attempts)")
            return True
        print(".", end="", flush=True)
        time.sleep(interval)

    print(f" FAILED after {max_attempts} attempts")
    return False


def main() -> int:
    """Wait for all services and return exit code."""
    print("=" * 50)
    print("Florida Tax RAG - Waiting for Docker Services")
    print("=" * 50)

    services = [
        ("Neo4j", check_neo4j),
        ("Weaviate", check_weaviate),
        ("Redis", check_redis),
    ]

    all_ready = True
    for name, check_fn in services:
        if not wait_for_service(name, check_fn):
            all_ready = False

    print("=" * 50)
    if all_ready:
        print("All services are ready!")
        return 0
    else:
        print("ERROR: Some services failed to start")
        return 1


if __name__ == "__main__":
    sys.exit(main())
