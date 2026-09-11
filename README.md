# 🎙️ Speech-to-Summary & Subtitle Transcriber

A high-performance Python & Gradio application that transcribes audio recordings (via file upload or native microphone streaming), generates precise sentence/time-segmented transcripts, creates downloadable `.srt` and `.vtt` subtitle files, and synthesizes structured executive meeting minutes using OpenAI LLMs.

---

## Demo

This project demonstrates an end-to-end speech processing workflow:

- Accepts audio input through a Gradio interface
- Transcribes speech using Whisper
- Generates an AI-powered summary
- Produces timestamped SRT/VTT subtitles
- Converts the transcript into structured meeting notes
- Provides an interactive visualization of the results

---

## 🚀 Key Features

* **Multi-Format Input:** Seamlessly upload standard audio formats (`.mp3`, `.wav`, `.m4a`) or record directly via a low-latency HTML5 microphone integration.
* **Granular Time-Segmented Transcriptions:** Utilizes Hugging Face's `Whisper` model configured with word/sentence boundary grouping to break long continuous recordings into clean 5–10 second timestamp blocks.
* **Automated Meeting Minutes:** Uses OpenAI API models to synthesize transcripts into structured summaries (Executive Summary, Discussion Points, Action Items, and Key Takeaways). Includes built-in safeguards to detect short audio clips and avoid summary hallucinations.
* **Subtitle Generation:** Automatically generates `.srt` and `.vtt` files ready for direct download.
* **Robust UI State Handling:** Built on Gradio 6.0+ with visual input resetting to avoid state collision between microphone and file inputs.

---

## 🛠️ Project Structure

```text
├── app.py           	# Gradio 6.0+ Web Interface, Custom JS/CSS, & Event Handlers
├── engine.py        	# Core Whisper pipeline execution, timestamp chunker, & OpenAI LLM summarizer
├── config.py        	# Global model settings, API key declarations, & threshold configs
├── visualizer.py    	# Optional audio feature rendering & visualization utilities
├── requirements.txt 	# Project dependency specifications
└── Readme.md        	# Technical documentation

```

---

## 📋 Prerequisites & Installation

### 1. Environment Setup

Ensure Python 3.10+ and `ffmpeg` are installed on your system.

```bash
# Clone the repository
git clone https://github.com/FMangiwa/speech-to-summary.git
cd speech-to-summary

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Environment Variables

Create a `.env` file or export your OpenAI API key:

```bash
export OPENAI_API_KEY="your-openai-api-key"

```

---

## ⚙️ How It Works

1. **Audio Ingestion (`app.py`):** Accepts files via `gr.Audio` or captures base64 audio streams using a native HTML5 JavaScript recorder interface.
2. **Transcription Engine (`engine.py`):** Passes input audio through Hugging Face's `openai/whisper-small` pipeline. Text is split into timestamped segments using sentence punctuation boundaries and 10-second windowing.
3. **Subtitle Parsing:** Converts segment timestamps into standard `.srt` (`00:00:00,000`) and `.vtt` (`00:00:00.000`) formats, saving them to temporary downloadable paths.
4. **LLM Summarization:** Evaluates word count; if valid, sends clean transcript text to OpenAI to generate Markdown-formatted meeting minutes.

---

## 🖥️ Usage

Run the Gradio app:

```bash
python app.py

```

Open the local server URL provided in the terminal (typically `http://127.0.0.1:7860`).

### User Interface Guide:

* **Upload Audio File:** Drop an existing recording to process.
* **Custom Mic Recorder:** Click **🔴 Start Recording**, record your message, click **⏹️ Stop Recording**, and listen to the instant audio preview.
* **🚀 Process Audio:** Executes transcription, subtitle generation, and LLM summarization.
* **🔄 Reset / Clear:** Clears Python state and JS DOM buffers, allowing clean switching between inputs without refreshing the browser.

---

## 🔧 Troubleshooting & Common Issues

* **Stale Audio Processing:** If the app processes an old microphone recording instead of a newly uploaded file, click **🔄 Reset / Clear** to wipe JavaScript state buffers.
* **Missing Subtitle Outputs:** Ensure system `ffmpeg` is properly configured in your system `PATH` if Whisper fails to process raw compressed audio formats.

---

## License

See `LICENSE`.

---