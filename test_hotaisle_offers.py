#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, "src")

from gpuhunt.providers.hotaisle import HotAisleProvider

# Get credentials from environment
api_key = os.getenv("HOTAISLE_API_KEY")
team_handle = os.getenv("HOTAISLE_TEAM_HANDLE")

if not api_key or not team_handle:
    print("Please set HOTAISLE_API_KEY and HOTAISLE_TEAM_HANDLE environment variables")
    sys.exit(1)

print(f"Testing Hotaisle provider with team_handle: {team_handle}")

# Create provider with credentials
provider = HotAisleProvider(api_key=api_key, team_handle=team_handle)

# Get offers
try:
    offers = list(provider.get())
    print(f"\nFound {len(offers)} offers:")

    for i, offer in enumerate(offers):
        print(f"\nOffer {i + 1}:")
        print(f"  Name: {offer.name}")
        print(f"  Price: ${offer.price}/hour")
        print(f"  Location: {offer.location}")
        print(f"  CPUs: {offer.cpu}")
        print(f"  Memory: {offer.memory} GB")
        print(f"  Disk: {offer.disk_size} GB")
        if hasattr(offer, "gpu_count") and offer.gpu_count > 0:
            print(f"  GPU Count: {offer.gpu_count}")
            print(f"  GPU Name: {offer.gpu_name}")
            print(f"  GPU Memory: {offer.gpu_memory} GB")
        print(f"  Full offer: {offer}")

except Exception as e:
    print(f"Error getting offers: {e}")
    import traceback

    traceback.print_exc()
