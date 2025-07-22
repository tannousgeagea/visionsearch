#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: waste_captions.py

Generates a full-paragraph waste-management caption for an image via
Moondream Cloud API. The caption describes the scene and focuses on the
user-provided labels as main objects.

Usage:
  python waste_captions.py --image /path/to/img.jpg [--labels label1,label2,...]

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

# Default categories if none provided
DEFAULT_LABELS = [
    "Plastics", "Cardboard", "Metal", "Cables/Cable Piles",
    "Wood", "Pipes", "Electronic Waste", "Mattresses", "Rigid Objects"
]
RETRY_COUNT = 3
RETRY_DELAY = 2  # seconds
MOONDREAM_KEY_ENV = "MOONDREAM_KEY"


def init_model(api_key: str):
    """Initialize and return a Moondream VL client."""
    return md.vl(api_key=api_key)


def build_prompt(labels):
    """Constructs the prompt to generate a full paragraph focusing on labels."""
    return (
        "Provide a single, detailed paragraph describing the entire scene, "
        "with emphasis on the following main objects: "
        f"{', '.join(labels)}. "
        "Describe their appearance, condition, and context within the scene. "
        "You may also mention any clearly visible cranes or machinery. "
        "Write in a professional, technical tone suitable for waste-to-energy risk assessment."
    )


def generate_caption(model, image, labels):
    """
    Calls the API with retries and returns the full generated caption.
    """
    prompt = build_prompt(labels)
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
    parser = argparse.ArgumentParser(description="Generate waste-management captions.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--labels", help="Comma-separated main object labels")
    args = parser.parse_args()

    api_key = os.getenv(MOONDREAM_KEY_ENV)
    if not api_key:
        print(f"Error: {MOONDREAM_KEY_ENV} not set.", file=sys.stderr)
        sys.exit(1)

    # Prepare labels list
    if args.labels:
        labels = [lbl.strip() for lbl in args.labels.split(',') if lbl.strip()]
    else:
        labels = DEFAULT_LABELS

    # Load image
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Error: could not open image: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Using main objects: {', '.join(labels)}\n")

    # Initialize model
    print("Initializing model...")
    model = init_model(api_key)
    print("Model ready.\n")

    # Generate and display caption
    print("Generating full-paragraph caption...")
    caption = generate_caption(model, image, labels)
    print(f"\nCaption:\n{caption}\n")

if __name__ == "__main__":
    main()
