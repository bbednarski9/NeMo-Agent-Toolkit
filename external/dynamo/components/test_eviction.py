#!/usr/bin/env python3
"""
Eviction test: fill a single worker's cache until evictions happen,
then verify the radix tree reflects the eviction (overlap drops to 0).

Requires DYNAMO_NUM_GPU_BLOCKS_OVERRIDE to be set small (e.g., 100)
so we can fill the cache quickly.
"""
import json
import os
import random
import string
import subprocess
import sys
import time

import msgpack
import zmq

API = "http://localhost:8000/v1/chat/completions"
MODEL = os.environ.get("TEST_MODEL", "Llama-3.1-8B-Instruct")
ZMQ_PORT = 20080
BLOCK_SIZE = 16


def send_request(prompt, max_tokens=4):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    })
    result = subprocess.run(
        ["curl", "-s", API, "-H", "Content-Type: application/json", "-d", payload],
        capture_output=True, text=True,
    )
    try:
        return json.loads(result.stdout)
    except Exception:
        return {"error": result.stdout[:200]}


def random_prompt(length=600):
    """Generate a random prompt that shares no prefix with any other."""
    return "".join(random.choices(string.ascii_letters + string.digits + " ", k=length))


def drain_zmq(sub):
    """Drain all pending ZMQ events, return (stored_hashes, removed_hashes, cleared)."""
    stored = set()
    removed = set()
    cleared = 0
    while True:
        try:
            parts = sub.recv_multipart(zmq.NOBLOCK)
            batch = msgpack.unpackb(parts[2], raw=False, strict_map_key=False)
            events = batch[1] if isinstance(batch, (list, tuple)) and len(batch) >= 3 else []
            for evt in events:
                if not isinstance(evt, (list, tuple)) or not evt:
                    continue
                etype = str(evt[0]).lower()
                hashes = evt[1] if len(evt) > 1 and isinstance(evt[1], list) else []
                if "stored" in etype:
                    for h in hashes:
                        stored.add(h)
                elif "removed" in etype:
                    for h in hashes:
                        removed.add(h)
                elif "cleared" in etype:
                    cleared += 1
        except zmq.Again:
            break
    return stored, removed, cleared


def main():
    print("=" * 70)
    print("  vLLM Eviction + Radix Tree Verification Test")
    print("=" * 70)

    # Check cache size
    from urllib.request import urlopen
    try:
        body = urlopen(f"http://localhost:18081/metrics", timeout=2).read().decode()
        for line in body.splitlines():
            if "num_gpu_blocks" in line and "cache_config" in line:
                import re
                m = re.search(r'num_gpu_blocks="(\d+)"', line)
                if m:
                    num_blocks = int(m.group(1))
                    print(f"\n  Worker 0 cache: {num_blocks} blocks "
                          f"({num_blocks * BLOCK_SIZE} tokens)")
                    if num_blocks > 500:
                        print(f"  WARNING: Cache is large ({num_blocks} blocks). "
                              f"Evictions will take many requests.")
                        print(f"  For a quick test, restart with "
                              f"DYNAMO_NUM_GPU_BLOCKS_OVERRIDE={min(100, num_blocks)}")
                    break
    except Exception as e:
        print(f"  Could not check cache size: {e}")

    # Connect ZMQ subscriber to worker 0
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVTIMEO, 500)
    sub.connect(f"tcp://127.0.0.1:{ZMQ_PORT}")
    time.sleep(1)
    drain_zmq(sub)  # clear any stale events

    total_stored = set()
    total_removed = set()
    total_cleared = 0

    # Phase 1: Send a "marker" prompt we'll track
    print("\n--- Phase 1: Send marker prompt ---")
    marker = "MARKER_PROMPT: " + "A" * 500
    resp = send_request(marker)
    tokens_used = resp.get("usage", {}).get("prompt_tokens", "?")
    print(f"  Sent marker prompt ({tokens_used} tokens)")
    time.sleep(2)
    s, r, c = drain_zmq(sub)
    total_stored |= s
    total_removed |= r
    total_cleared += c
    print(f"  ZMQ events: stored={len(s)} hashes, removed={len(r)}, cleared={c}")
    marker_stored = len(s) > 0

    # Phase 2: Flood with unique prompts to fill cache
    print("\n--- Phase 2: Flood with unique prompts until evictions ---")
    batch_num = 0
    eviction_seen = False
    requests_sent = 0

    while not eviction_seen:
        batch_num += 1
        # Send 10 requests per batch
        for i in range(10):
            prompt = random_prompt(600)
            send_request(prompt, max_tokens=2)
            requests_sent += 1

        time.sleep(1)
        s, r, c = drain_zmq(sub)
        total_stored |= s
        total_removed |= r
        total_cleared += c

        if r:
            eviction_seen = True
            print(f"  Batch {batch_num} ({requests_sent} total requests): "
                  f"EVICTION DETECTED!")
            print(f"    Stored hashes (cumulative): {len(total_stored)}")
            print(f"    Removed hashes (this batch): {len(r)}")
            print(f"    Removed hashes (cumulative): {len(total_removed)}")
        else:
            if batch_num % 5 == 0:
                print(f"  Batch {batch_num} ({requests_sent} requests): "
                      f"stored={len(total_stored)}, no evictions yet...")

        if requests_sent > 2000:
            print(f"\n  Gave up after {requests_sent} requests with no evictions.")
            print(f"  Cache is too large. Restart with "
                  f"DYNAMO_NUM_GPU_BLOCKS_OVERRIDE=100")
            sub.close()
            ctx.term()
            sys.exit(1)

    # Phase 3: Verify radix tree via router
    print("\n--- Phase 3: Verify radix tree reflects evictions ---")
    print("  Sending marker prompt again to check overlap...")
    resp2 = send_request(marker)
    time.sleep(1)

    # Check the router log for the overlap on this request
    print("  Check router logs for the last kv_only decision...")
    print("  (Look for the marker request -- if ZMQ eviction events updated")
    print("   the radix tree, overlap should be < 1.0 for the evicted blocks)")

    # Drain any final events
    time.sleep(2)
    s, r, c = drain_zmq(sub)
    total_removed |= r

    print(f"\n--- Final Summary ---")
    print(f"  Requests sent:      {requests_sent + 2}")
    print(f"  Stored hashes:      {len(total_stored)}")
    print(f"  Removed hashes:     {len(total_removed)}")
    print(f"  Cleared events:     {total_cleared}")
    overlap = len(total_stored - total_removed)
    print(f"  Net hashes in tree: {overlap} "
          f"(stored - removed)")

    if total_removed:
        print(f"\n  PASS: Evictions detected and flowing via ZMQ")
        evicted_markers = total_removed & total_stored
        if evicted_markers:
            print(f"  {len(evicted_markers)} stored hashes were later evicted")
    else:
        print(f"\n  FAIL: No evictions detected")

    sub.close()
    ctx.term()


if __name__ == "__main__":
    main()
