# English Accent Analyzer

A tool that analyzes English accents from videos to help evaluate spoken English for hiring purposes.

## Features

- Accepts public video URLs (YouTube, Loom, or direct MP4 links)
- Extracts audio from videos
- Analyzes the speaker's accent to detect English language speaking candidates
- Outputs:
  - Classification of the accent (British, American, Australian, etc.)
  - Confidence in English accent score (0-100%)
  - Detailed explanation of the accent detection

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Some additional system dependencies might be required for audio processing:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg libavcodec-extra

# macOS
brew install ffmpeg
```

## Usage

### Command Line Interface

```bash
# Analyze a video from a URL
python accent_analyzer.py --url "https://www.youtube.com/watch?v=example"

# Launch the web interface
python accent_analyzer.py --web
```

### Web Interface

The tool provides a Streamlit web interface that can be accessed at http://localhost:8501 after launching with the `--web` flag.

## How It Works

1. **Video Processing**:
   - Downloads video from the provided URL
   - Extracts audio track from the video

2. **Audio Analysis**:
   - Uses Whisper for initial speech recognition and language detection
   - Applies Wav2Vec2 for detailed phoneme analysis
   - Extracts acoustic features (pitch, rhythm, formants)

3. **Accent Classification**:
   - Analyzes phoneme patterns that distinguish different English accents
   - Examines prosodic features (intonation, rhythm)
   - Classifies based on a combination of phonetic and acoustic patterns

4. **Result Generation**:
   - Provides accent classification with confidence score
   - Generates human-readable explanation
   - Confirms if the speech is in English

## Accent Detection Methodology

The system analyzes several key features to distinguish between different English accents:

- **Rhotic vs. Non-Rhotic**: Whether 'r' sounds are pronounced after vowels
- **T-Flapping**: How 't' sounds are pronounced (clearly or more like 'd')
- **Vowel Quality**: The specific sounds used for vowels
- **Intonation Patterns**: Rising or falling pitch at the end of sentences
- **Rhythm**: Stress-timed vs. syllable-timed speech patterns

## Limitations

- Best results are achieved with clear audio and minimal background noise
- The system is designed for English language detection and may not accurately classify non-English speech
- Accent detection is probabilistic and based on general patterns
- Mixed accents may be classified as the closest matching category

## Deployment

For production deployment, consider:

1. **Containerization**:
   ```bash
   docker build -t accent-analyzer .
   docker run -p 8501:8501 accent-analyzer
   ```

2. **Cloud Deployment**:
   - Deploy to Heroku, Google Cloud Run, or AWS Elastic Beanstalk
   - Set up appropriate authentication for production use

## License

This project is licensed under the MIT License - see the LICENSE file for details.