import os
import sys
import base64
import tempfile
import importlib
import gradio as gr

if "engine" in sys.modules:
    importlib.reload(sys.modules["engine"])

import engine

os.environ["GRADIO_TEMP_DIR"] = "/tmp/gradio"

def process_pipeline(file_audio, mic_audio_b64):
    audio_path = None

    # Check uploaded file first, fallback to mic recording if no file present
    if file_audio:
        audio_path = file_audio
    elif mic_audio_b64 and isinstance(mic_audio_b64, str) and len(mic_audio_b64.strip()) > 100:
        try:
            raw_b64 = mic_audio_b64.strip()
            if "," in raw_b64:
                header, encoded = raw_b64.split(",", 1)
            else:
                encoded = raw_b64
                
            data = base64.b64decode(encoded)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(data)
            temp_file.close()
            audio_path = temp_file.name
        except Exception as e:
            return f"Error decoding audio stream: {e}", "", "", "", gr.update(visible=False), gr.update(visible=False)

    if not audio_path:
        return "No audio provided.", "", "", "", gr.update(visible=False), gr.update(visible=False)

    readable_text, srt_sub, vtt_sub, srt_file, vtt_file = engine.transcribe_audio(audio_path)
    summary = engine.generate_minutes(readable_text)

    return (
        summary,
        readable_text,
        srt_sub,
        vtt_sub,
        gr.update(value=srt_file, visible=True),
        gr.update(value=vtt_file, visible=True),
    )

def clear_all_inputs():
    """Python handler to reset Python-side components."""
    return (
        None,                        # Resets file_input
        "",                          # Resets mic_b64_input
        "",                          # Clears summary_output
        "",                          # Clears transcript_output
        "",                          # Clears srt_output
        "",                          # Clears vtt_output
        gr.update(visible=False),    # Hides srt_download button
        gr.update(visible=False),    # Hides vtt_download button
    )

HEAD_JS = """
<script>
let mediaRecorder;
let audioChunks = [];

function clearMicDOM() {
    const gradioContainer = document.getElementById('mic_b64_output');
    const gradioInput = gradioContainer ? gradioContainer.querySelector('textarea') : null;
    if (gradioInput) {
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
        nativeInputValueSetter.call(gradioInput, "");
        gradioInput.dispatchEvent(new Event('input', { bubbles: true }));
        gradioInput.dispatchEvent(new Event('change', { bubbles: true }));
    }
    const statusLabel = document.getElementById('mic_status_label');
    if (statusLabel) {
        statusLabel.innerText = "Status: Ready to test microphone";
        statusLabel.style.color = "#666";
    }
    const preview = document.getElementById('audio_preview');
    if (preview) {
        preview.src = "";
    }
}

async function startRecording() {
    audioChunks = [];
    const statusLabel = document.getElementById('mic_status_label');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            document.getElementById('audio_preview').src = audioUrl;
            
            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            reader.onloadend = () => {
                const base64Audio = reader.result;
                const gradioContainer = document.getElementById('mic_b64_output');
                const gradioInput = gradioContainer ? gradioContainer.querySelector('textarea') : null;
                
                if (gradioInput) {
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                    nativeInputValueSetter.call(gradioInput, base64Audio);
                    
                    gradioInput.dispatchEvent(new Event('input', { bubbles: true }));
                    gradioInput.dispatchEvent(new Event('change', { bubbles: true }));
                }
            };
            statusLabel.innerText = "Status: ✅ Recording captured! Ready for processing.";
            statusLabel.style.color = "green";
        };
        
        mediaRecorder.start();
        document.getElementById('start_rec_btn').disabled = true;
        document.getElementById('stop_rec_btn').disabled = false;
        statusLabel.innerText = "Status: 🔴 Recording in progress...";
        statusLabel.style.color = "red";
    } catch (err) {
        statusLabel.innerText = "Error: " + err.message;
        statusLabel.style.color = "red";
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        document.getElementById('start_rec_btn').disabled = false;
        document.getElementById('stop_rec_btn').disabled = true;
    }
}
</script>
"""

CUSTOM_MIC_HTML = """
<div style="border: 1px solid #ccc; padding: 15px; border-radius: 8px; background: #f9f9f9;">
    <h3>🎤 Native HTML5 Microphone Recorder</h3>
    <p id="mic_status_label" style="color: #666;">Status: Ready to test microphone</p>
    <button id="start_rec_btn" onclick="startRecording()" style="padding: 8px 16px; margin-right: 8px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">🔴 Start Recording</button>
    <button id="stop_rec_btn" onclick="stopRecording()" disabled style="padding: 8px 16px; background-color: #dc3545; color: white; border: none; border-radius: 4px; cursor: pointer;">⏹️ Stop Recording</button>
    <audio id="audio_preview" controls style="display: block; margin-top: 10px; width: 100%;"></audio>
</div>
"""

CUSTOM_CSS = """
.scrollable-box textarea {
    height: 380px !important;
    max-height: 380px !important;
    overflow-y: auto !important;
    white-space: pre-wrap !important;
}

#mic_b64_output {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
    height: 0px !important;
    overflow: hidden !important;
}
"""

with gr.Blocks(title="Speech-to-Summary & Subtitle Transcriber") as demo:
    gr.Markdown("# 🎙️ Speech-to-Summary & Subtitle Transcriber")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tab("📁 Upload Audio File"):
                file_input = gr.Audio(sources=["upload"], type="filepath", label="Audio File")

            with gr.Tab("🎤 Custom Mic Recorder"):
                gr.HTML(CUSTOM_MIC_HTML)
                mic_b64_input = gr.Textbox(elem_id="mic_b64_output", visible=True)

            # Placed cleanly as main column controls
            submit_btn = gr.Button("🚀 Process Audio", variant="primary")
            reset_btn = gr.Button("🔄 Reset / Clear", variant="secondary")

        with gr.Column(scale=2):
            with gr.Tab("📋 Meeting Minutes Summary"):
                summary_output = gr.Textbox(
                    label="Structured Summary", 
                    elem_classes=["scrollable-box"]
                )

            with gr.Tab("📖 Readable Transcript"):
                transcript_output = gr.Textbox(
                    label="Clean Text", 
                    elem_classes=["scrollable-box"]
                )

            with gr.Tab("⏱️ SRT Subtitles"):
                srt_output = gr.Textbox(
                    label=".srt Format Timestamps", 
                    elem_classes=["scrollable-box"]
                )
                srt_download = gr.File(label="Download .srt File", visible=False)

            with gr.Tab("⏱️ VTT Subtitles"):
                vtt_output = gr.Textbox(
                    label=".vtt Format Timestamps", 
                    elem_classes=["scrollable-box"]
                )
                vtt_download = gr.File(label="Download .vtt File", visible=False)

    submit_btn.click(
        fn=process_pipeline,
        inputs=[file_input, mic_b64_input],
        outputs=[
            summary_output,
            transcript_output,
            srt_output,
            vtt_output,
            srt_download,
            vtt_download,
        ],
    )

    reset_btn.click(
        fn=clear_all_inputs,
        inputs=None,
        outputs=[
            file_input,
            mic_b64_input,
            summary_output,
            transcript_output,
            srt_output,
            vtt_output,
            srt_download,
            vtt_download,
        ],
        js="() => { clearMicDOM(); }"
    )

if __name__ == "__main__":
    demo.launch(share=True, head=HEAD_JS, css=CUSTOM_CSS)