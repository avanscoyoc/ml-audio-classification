"""Entry point for running ml_audio_classification as a module."""

import asyncio
from .cli import main

if __name__ == "__main__":
    asyncio.run(main())