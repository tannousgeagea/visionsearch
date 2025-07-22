#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: describe_image.py

Generates a detailed, professional paragraph describing the entire scene of an image via
Moondream Cloud API. Focuses purely on visible objects and context, without requiring labels.

Usage:
  python describe_image.py --image /path/to/img.jpg

Configure:
  export MOONDREAM_KEY="your-api-key"

Requirements:
  pip install moondream pillow
"""
import os
import sys
import time
import argparse
from PIL import Image
import moondream as md
import urllib.error

RETRY_COUNT = 3
RETRY_DELAY = 2  # seconds
MOONDREAM_KEY_ENV = "MOONDREAM_KEY"


def init_model(api_key: str):
    """Initialize and return a Moondream VL client."""
    return md.vl(api_key=api_key)


def build_prompt():
    """Constructs the prompt to describe the entire scene."""
    return (
        "Provide a single, detailed paragraph describing all visible objects and their context within the scene. "
        "Identify key machinery, structures, and materials present. "
        "Highlight any potential hazards or noteworthy components. "
        "Write in a professional, technical tone suitable for waste-to-energy risk assessment."
    )


def generate_description(model, image):
    """
    Calls the API with retries and returns the full generated description.
    """
    prompt = build_prompt()
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = model.query(image, prompt)
            return resp.get("answer", resp.get("caption", "")).strip()
        except urllib.error.HTTPError as e:
            print(f"Warning: HTTP {e.code} (attempt {attempt}/{RETRY_COUNT}), retrying...", file=sys.stderr)
            time.sleep(RETRY_DELAY)
    # Final attempt without catch
    resp = model.query(image, prompt)
    return resp.get("answer", resp.get("caption", "")).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate a detailed image description.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    api_key = os.getenv(MOONDREAM_KEY_ENV)
    if not api_key:
        print(f"Error: {MOONDREAM_KEY_ENV} not set.", file=sys.stderr)
        sys.exit(1)

    # Load image
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Error: could not open image: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize model
    print("Initializing model...")
    model = init_model(api_key)
    print("Model ready.\n")

    # Generate and display description
    print("Generating description...")
    description = generate_description(model, image)
    print(f"\nDescription:\n{description}\n")

if __name__ == "__main__":
    main()
