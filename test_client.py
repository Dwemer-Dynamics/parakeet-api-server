#!/usr/bin/env python3
"""
Simple test client for Parakeet STT API
"""
import argparse
import sys
import requests
from pathlib import Path


def test_transcription(audio_file: str, server_url: str = "http://localhost:8022"):
    """
    Test transcription endpoint

    Args:
        audio_file: Path to audio file
        server_url: Server URL
    """
    audio_path = Path(audio_file)

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    print(f"Testing Parakeet STT API")
    print(f"Server: {server_url}")
    print(f"Audio: {audio_file}")
    print("-" * 60)

    # Test health endpoint
    try:
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{server_url}/health")
        response.raise_for_status()
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   Model: {health['model']}")
        print(f"   Precision: {health['precision']}")
        print(f"   Backend: {health['backend']}")
    except Exception as e:
        print(f"   Error: {e}")
        print("\nMake sure the server is running!")
        sys.exit(1)

    # Test transcription endpoint
    try:
        print("\n2. Testing transcription endpoint...")
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            response = requests.post(f"{server_url}/v1/transcribe", files=files)
            response.raise_for_status()

        result = response.json()
        print(f"   Transcription: {result['text']}")
    except Exception as e:
        print(f"   Error: {e}")
        sys.exit(1)

    # Test OpenAI-compatible endpoint
    try:
        print("\n3. Testing OpenAI-compatible endpoint...")
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            data = {"model": "parakeet-tdt-0.6b-v3"}
            response = requests.post(
                f"{server_url}/v1/audio/transcriptions",
                files=files,
                data=data
            )
            response.raise_for_status()

        result = response.json()
        print(f"   Transcription: {result['text']}")
    except Exception as e:
        print(f"   Error: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test Parakeet STT API")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument(
        "--server",
        default="http://localhost:8022",
        help="Server URL (default: http://localhost:8022)"
    )

    args = parser.parse_args()
    test_transcription(args.audio_file, args.server)


if __name__ == "__main__":
    main()
