"""ClipForge local pipeline — MoneyPrinterTurbo heritage.

Service modules:
- task: pipeline orchestration (in-process worker thread)
- download: yt-dlp fetcher + upload ingest
- llm: Gemini clip detection
- subtitle: WhisperX (or batched faster-whisper) transcription + ASS caption authoring
- video: ffmpeg/MoviePy 9:16 rendering, face reframe, silence trim
- vad: Silero VAD silence detection
- reframe: MediaPipe face-tracking crop windows
"""

__version__ = "1.0.0"
