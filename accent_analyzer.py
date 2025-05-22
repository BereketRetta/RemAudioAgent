import os
import sys
import tempfile
import argparse
import warnings
import json
from pydub import AudioSegment
import requests
import streamlit as st
import numpy as np
import librosa
import pytube
import re
import yt_dlp
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (contains OPENAI_API_KEY)
load_dotenv()


class AccentAnalyzerOpenAI:
    """
    A tool to analyze English accents from videos by extracting audio
    and using OpenAI's API for speech recognition and analysis.
    """

    # Define accent categories and their characteristics
    ACCENT_PROFILES = {
        "American": {
            "description": "American English is characterized by rhotic pronunciation (pronouncing 'r' sounds), 't' flapping, and distinctive vowel sounds.",
            "key_features": ["rhotic 'r'", "t-flapping", "nasal 'a'", "short vowels"]
        },
        "British": {
            "description": "British English (RP) is characterized by non-rhotic pronunciation, clear 't' sounds, and distinctive vowel qualities.",
            "key_features": ["non-rhotic", "clear 't'", "rounded 'o'", "glottal stops"]
        },
        "Australian": {
            "description": "Australian English has distinctive vowel shifts, is non-rhotic, and has a notable rising intonation.",
            "key_features": ["rising intonation", "non-rhotic", "distinctive 'i' sound", "vowel shifts"]
        },
        "Indian": {
            "description": "Indian English has a syllable-timed rhythm, retroflex consonants, and distinctive vowel qualities.",
            "key_features": ["retroflex consonants", "syllable-timed rhythm", "stress patterns", "vowel distinctions"]
        },
        "Canadian": {
            "description": "Canadian English shares features with American English but has distinctive vowel sounds and 'Canadian raising'.",
            "key_features": ["Canadian raising", "rhotic", "distinctive 'about'", "mix of US/UK features"]
        },
        "Irish": {
            "description": "Irish English has a melodic intonation, distinctive vowel qualities, and a mix of rhotic and non-rhotic features.",
            "key_features": ["melodic intonation", "distinctive vowels", "soft consonants", "rhythmic patterns"]
        }
    }

    # Word patterns that are characteristic of different accents (for additional analysis)
    ACCENT_WORD_PATTERNS = {
        "American": [
            r"\b(gonna|wanna|gotta)\b",  # Contracted forms
            r"\b(awesome|cool|totally)\b",  # Common American slang
            r"\b(apartment|sidewalk|elevator|subway|vacation)\b",  # American terms
            r"\b(color|center|gray|theater)\b",  # American spelling
        ],
        "British": [
            r"\b(flat|lift|lorry|boot|bonnet|rubbish|queue)\b",  # British terms
            r"\b(colour|centre|grey|theatre)\b",  # British spelling
            # Common British expressions
            r"\b(rather|quite|proper|brilliant|lovely)\b",
            r"\b(whilst|amongst)\b",  # British preference
        ],
        "Australian": [
            r"\b(mate|g'day|arvo|barbie|brekkie)\b",  # Australian slang
            r"\b(reckon|crikey|fair dinkum)\b",  # Australian expressions
            r"\b(thongs|ute|esky|servo)\b",  # Australian terms
        ],
        "Indian": [
            r"\b(yaar|acha|thik hai)\b",  # Indian expressions
            r"\b(itself|only|na|no)\b",  # Characteristic usage
            r"\b(doing the needful|prepone)\b",  # Indian English phrases
        ],
        "Canadian": [
            r"\b(eh|toque|loonie|toonie)\b",  # Canadian terms
            # Canadian vocabulary
            r"\b(washroom|hydro|provinces|poutine|chesterfield)\b",
            # Words with distinctive Canadian pronunciation
            r"\b(sorry|about)\b",
        ],
        "Irish": [
            r"\b(grand|savage|gas|craic)\b",  # Irish slang
            r"\b(fierce|deadly|sound)\b",  # Irish usage
            r"\b(eejit|culchie|yoke)\b",  # Irish terms
        ]
    }

    def __init__(self, api_key=None):
        # Initialize OpenAI client
        self.openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as a parameter.")

        self.client = OpenAI(api_key=self.openai_api_key)

    def download_audio_only(self, url):
        """Download only the audio from a video URL and return the path to the downloaded file."""
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.mp3")

        st.write(f"Downloading audio from {url}...")

        if "youtube" in url or "youtu.be" in url:
            # Use yt-dlp to download only audio from YouTube
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, 'audio'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                # yt-dlp might add extensions, find the downloaded file
                for file in os.listdir(temp_dir):
                    if file.endswith(".mp3"):
                        audio_path = os.path.join(temp_dir, file)
                        break
            except Exception as e:
                # Fallback to pytube if yt-dlp fails
                st.write(f"yt-dlp failed, falling back to pytube: {e}")
                yt = pytube.YouTube(url)
                audio_stream = yt.streams.filter(only_audio=True).first()
                downloaded_file = audio_stream.download(output_path=temp_dir)

                # Convert to mp3
                audio = AudioSegment.from_file(downloaded_file)
                audio.export(audio_path, format="mp3")

        elif "loom.com" in url:
            # Handle Loom videos by extracting the direct audio URL if possible
            try:
                response = requests.get(url)
                # First try to find direct audio URL
                audio_match = re.search(r'"audioUrl":"([^"]+)"', response.text)

                if audio_match:
                    # Direct audio URL found
                    audio_url = audio_match.group(1).replace('\\', '')
                    audio_response = requests.get(audio_url, stream=True)
                    with open(audio_path, 'wb') as f:
                        for chunk in audio_response.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                else:
                    # Fallback to video URL and extract audio
                    video_match = re.search(
                        r'"playbackUrl":"([^"]+)"', response.text)
                    if video_match:
                        video_url = video_match.group(1).replace('\\', '')
                        # Use yt-dlp to extract audio only
                        ydl_opts = {
                            'format': 'bestaudio/best',
                            'outtmpl': os.path.join(temp_dir, 'audio'),
                            'postprocessors': [{
                                'key': 'FFmpegExtractAudio',
                                'preferredcodec': 'mp3',
                                'preferredquality': '192',
                            }],
                        }
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([video_url])

                        # Find the downloaded file
                        for file in os.listdir(temp_dir):
                            if file.endswith(".mp3"):
                                audio_path = os.path.join(temp_dir, file)
                                break
                    else:
                        raise ValueError(
                            "Could not extract audio or video URL from Loom page")
            except Exception as e:
                st.error(f"Error processing Loom video: {e}")
                raise

        elif url.endswith((".mp3", ".wav", ".m4a", ".ogg")):
            # Already an audio file, just download it
            response = requests.get(url, stream=True)
            with open(audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        elif url.endswith((".mp4", ".mov", ".avi", ".webm")):
            # Handle direct video links by downloading and extracting audio
            temp_video_path = os.path.join(temp_dir, "temp_video")
            response = requests.get(url, stream=True)
            with open(temp_video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            # Extract audio using pydub
            video = AudioSegment.from_file(temp_video_path)
            video.export(audio_path, format="mp3")

            # Clean up the temporary video file
            try:
                os.remove(temp_video_path)
            except:
                pass
        else:
            raise ValueError(
                "Unsupported URL format. Please provide a YouTube, Loom, or direct audio/video link.")

        st.write(f"Audio downloaded to {audio_path}")
        return audio_path

    def convert_audio_for_openai(self, audio_path):
        """Convert audio to format compatible with OpenAI API (mp3)."""
        filename, ext = os.path.splitext(audio_path)
        if ext.lower() == '.mp3':
            return audio_path

        # Convert to mp3 if it's not already
        mp3_path = f"{filename}.mp3"
        audio = AudioSegment.from_file(audio_path)
        audio.export(mp3_path, format="mp3")
        return mp3_path

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using OpenAI's Whisper API.
        Returns the transcription and language detection.
        """
        st.write("Transcribing audio with OpenAI's Whisper API...")

        # Ensure audio is in compatible format
        mp3_path = self.convert_audio_for_openai(audio_path)

        try:
            with open(mp3_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )

            # Parse the JSON response
            if isinstance(response, str):
                result = json.loads(response)
            else:
                # If already parsed as an object
                result = response

            transcription = result.text if hasattr(
                result, 'text') else result.get('text', '')
            language = result.language if hasattr(
                result, 'language') else result.get('language', '')

            return {
                "transcription": transcription,
                "language": language,
                "is_english": language.lower() == "en" or language.lower() == "english"
            }
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            # Fallback - return empty results
            return {
                "transcription": "",
                "language": "unknown",
                "is_english": False,
                "error": str(e)
            }

    def analyze_accent_with_openai(self, transcription):
        """
        Use OpenAI's GPT model to analyze the accent based on transcription.
        """
        st.write("Analyzing accent with GPT...")

        try:
            # Create a prompt that asks GPT to analyze the accent
            prompt = f"""
            Analyze the following English speech transcription and determine the likely accent of the speaker. 
            Consider word choices, spelling patterns, and any distinctive phrases that might indicate a specific English accent.
            
            The possible accents are: American, British, Australian, Indian, Canadian, or Irish.
            
            Please provide:
            1. The most likely accent
            2. A confidence percentage (0-100%)
            3. A brief explanation of why you determined this accent
            
            Format your response as a JSON object with the following fields:
            - accent: The determined accent
            - confidence: The confidence percentage (a number, not a string)
            - explanation: Your explanation
            
            Transcription:
            "{transcription}"
            
            Respond with ONLY the JSON object, no additional text:
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are an expert linguist specializing in accent analysis."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            # Extract and parse the JSON response
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)

            return result

        except Exception as e:
            st.error(f"Error analyzing accent with GPT: {e}")
            return {
                "accent": "Unknown",
                "confidence": 0,
                "explanation": f"Error analyzing accent: {str(e)}"
            }

    def extract_audio_features(self, audio_path):
        """Extract acoustic features from audio for additional accent analysis."""
        st.write("Extracting acoustic features...")

        try:
            # Load audio
            audio, sample_rate = librosa.load(audio_path, sr=16000)

            # Extract rhythm metrics
            onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
            tempo, _ = librosa.beat.beat_track(
                onset_envelope=onset_env, sr=sample_rate)

            # Extract pitch information
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
            pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)])

            # Extract speech rate (approximated by onsets per second)
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env, sr=sample_rate)
            speech_rate = len(onset_frames) / (len(audio) / sample_rate)

            return {
                "tempo": tempo,
                "pitch_mean": pitch_mean if not np.isnan(pitch_mean) else 0,
                "speech_rate": speech_rate
            }
        except Exception as e:
            st.warning(f"Could not extract all acoustic features: {e}")
            return {
                "tempo": 120,  # Default values
                "pitch_mean": 0,
                "speech_rate": 4
            }

    def analyze_word_patterns(self, transcription):
        """Analyze the transcription for word patterns characteristic of different accents."""
        scores = {}

        # Convert to lowercase for pattern matching
        text = transcription.lower()

        # Calculate matches for each accent's patterns
        for accent, patterns in self.ACCENT_WORD_PATTERNS.items():
            match_count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                match_count += len(matches)

            # Normalize by the number of patterns
            scores[accent] = match_count / len(patterns) if patterns else 0

        # Normalize all scores relative to the maximum
        max_score = max(scores.values()) if scores and max(
            scores.values()) > 0 else 1
        for accent in scores:
            scores[accent] = scores[accent] / max_score

        return scores

    def enhance_accent_analysis(self, openai_analysis, word_patterns, acoustic_features):
        """
        Enhance the GPT-based accent analysis with additional acoustic and lexical features.
        This can sometimes help refine the analysis, especially for borderline cases.
        """
        accent = openai_analysis["accent"]
        confidence = openai_analysis["confidence"]
        explanation = openai_analysis["explanation"]

        # If confidence is already high, don't modify much
        if confidence > 85:
            return openai_analysis

        # Get word pattern score for the detected accent
        pattern_score = word_patterns.get(accent, 0)

        # Check if another accent has a much higher word pattern score
        alternative_accent = None
        for acc, score in word_patterns.items():
            if acc != accent and score > pattern_score * 1.5 and score > 0.5:
                alternative_accent = acc
                pattern_score = score

        # If acoustic features strongly suggest a different accent, consider it
        if acoustic_features:
            # Example: faster speech rate often associated with American accent
            if accent != "American" and acoustic_features["speech_rate"] > 5.5 and pattern_score < 0.3:
                alternative_accent = "American"

            # Example: specific pitch patterns for Australian/Irish accents
            if (accent not in ["Australian", "Irish"]) and acoustic_features["pitch_mean"] > 200 and pattern_score < 0.3:
                if word_patterns.get("Australian", 0) > word_patterns.get("Irish", 0):
                    alternative_accent = "Australian"
                else:
                    alternative_accent = "Irish"

        # If we found a compelling alternative, adjust the analysis
        if alternative_accent and alternative_accent != accent:
            original_confidence = confidence
            # Reduce confidence
            confidence = max(confidence * 0.7, min(confidence, 75))

            # Add note about the uncertainty
            explanation += f"\n\nNote: There are also features consistent with a {alternative_accent} accent. The confidence has been adjusted from {original_confidence}% to {confidence}% to reflect this uncertainty."

            # In some cases, switch to the alternative if evidence is strong
            if pattern_score > 0.8 and confidence < 60:
                accent = alternative_accent
                confidence = min(pattern_score * 70, 75)  # Cap at 75%
                explanation += f"\n\nBased on word choice patterns, the accent has been reclassified as {accent}."

        return {
            "accent": accent,
            "confidence": confidence,
            "explanation": explanation
        }

    def process_audio(self, audio_path):
        """
        Process an audio file and return accent analysis results.
        """
        try:
            # Convert to format compatible with OpenAI if needed
            mp3_path = self.convert_audio_for_openai(audio_path)

            # Transcribe the audio
            transcription_result = self.transcribe_audio(mp3_path)
            transcription = transcription_result["transcription"]
            is_english = transcription_result["is_english"]

            # If it's not English, return that info
            if not is_english:
                return {
                    "accent": "Non-English",
                    "confidence": 90,
                    "explanation": f"The speech appears to be in {transcription_result['language']} rather than English.",
                    "is_english": False,
                    "transcription": transcription
                }

            # Analyze the accent using GPT
            gpt_analysis = self.analyze_accent_with_openai(transcription)

            # Additional analyses to enhance the results
            word_patterns = self.analyze_word_patterns(transcription)
            acoustic_features = self.extract_audio_features(mp3_path)

            # Combine analyses for final result
            enhanced_analysis = self.enhance_accent_analysis(
                gpt_analysis, word_patterns, acoustic_features)

            # Add transcription to the result
            enhanced_analysis["transcription"] = transcription
            enhanced_analysis["is_english"] = True

            return enhanced_analysis

        except Exception as e:
            st.error(f"Error in accent analysis: {e}")
            return {
                "accent": "Error",
                "confidence": 0,
                "explanation": f"Error analyzing accent: {str(e)}",
                "is_english": False,
                "transcription": ""
            }

    def process_video(self, url):
        """
        Process a video URL and return accent analysis results.
        """
        try:
            # Download only the audio
            audio_path = self.download_audio_only(url)

            # Analyze the audio
            result = self.process_audio(audio_path)

            # Clean up temporary files
            try:
                os.remove(audio_path)
            except:
                pass

            return result

        except Exception as e:
            st.error(f"Error processing video: {e}")
            return {
                "error": str(e),
                "accent": "Unknown",
                "confidence": 0,
                "explanation": f"Error processing video: {str(e)}",
                "is_english": False,
                "transcription": ""
            }


def create_streamlit_app():
    """Create a Streamlit web app interface."""
    st.set_page_config(page_title="English Accent Analyzer",
                       page_icon="🎤", layout="wide")

    st.title("English Accent Analyzer")
    st.markdown("""
    This tool analyzes the English accent of speakers in videos.
    """)
    # URL input
    url = st.text_input(
        "Enter a video URL (YouTube, Loom, or direct audio/video link):")

    # File upload option
    uploaded_file = st.file_uploader("Or upload an audio/video file:",
                                     type=["mp3", "wav", "mp4", "m4a", "mov", "avi"])

    if st.button("Analyze Accent"):
        if not os.getenv("OPENAI_API_KEY"):
            st.error(
                "OpenAI API key is required. Please provide it above or set the OPENAI_API_KEY environment variable.")
        elif not url and not uploaded_file:
            st.error("Please provide either a URL or upload a file.")
        else:
            try:
                with st.spinner("Initializing analyzer..."):
                    analyzer = AccentAnalyzerOpenAI(api_key=None)

                if url:
                    with st.spinner("Processing video from URL..."):
                        result = analyzer.process_video(url)
                        display_results(result)

                elif uploaded_file:
                    with st.spinner("Processing uploaded file..."):
                        # Save uploaded file temporarily
                        temp_dir = tempfile.mkdtemp()
                        file_extension = os.path.splitext(
                            uploaded_file.name)[1].lower()
                        temp_path = os.path.join(
                            temp_dir, f"uploaded_file{file_extension}")

                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.read())

                        # Process the audio
                        result = analyzer.process_audio(temp_path)
                        display_results(result)

                        # Clean up
                        try:
                            os.remove(temp_path)
                        except:
                            pass
            except Exception as e:
                st.error(f"Error: {str(e)}")


def display_results(result):
    """Display accent analysis results in the Streamlit UI."""
    if "error" in result and result["error"]:
        st.error(f"Error: {result['error']}")
        return

    st.success("Analysis complete!")

    # Create columns for layout
    col1, col2 = st.columns(2)

    with col1:
        # Display accent classification
        st.subheader("Accent Classification")
        st.markdown(
            f"<h3 style='color:#1E88E5;'>{result['accent']}</h3>", unsafe_allow_html=True)

        # Display confidence score with gauge chart
        st.subheader("Confidence Score")
        score = min(100, max(0, result['confidence']))

        # Create a color based on confidence
        if score > 80:
            color = "#4CAF50"  # Green
        elif score > 60:
            color = "#FFA726"  # Orange
        else:
            color = "#EF5350"  # Red

        # Display progress bar and percentage
        st.progress(score / 100)
        st.markdown(
            f"<h4 style='color:{color};'>{score:.1f}%</h4>", unsafe_allow_html=True)

    with col2:
        # Display transcription
        st.subheader("Transcription")
        st.text_area("", value=result.get('transcription',
                     'No transcription available'), height=175)

    # Display explanation
    st.subheader("Analysis Explanation")
    st.write(result['explanation'])


def main():
    """Main entry point for the application."""
    create_streamlit_app()


if __name__ == "__main__":
    main()
