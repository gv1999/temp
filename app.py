from flask import Flask, request, jsonify
from utils import download_audio, transcribe_audio, summarize_text_with_t5

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize_youtube_video():
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    
    if not youtube_url:
        return jsonify({"error": "YouTube URL is required"}), 400

    # Download audio from YouTube
    audio_file = download_audio(youtube_url)
    if audio_file is None:
        return jsonify({"error": "Failed to download audio."}), 500

    # Transcribe audio
    transcript = transcribe_audio(audio_file)

    # Summarize transcript
    summary = summarize_text_with_t5(transcript)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
