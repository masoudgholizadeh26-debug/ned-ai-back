import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app) # Enable CORS for all origins

# Configure Generative AI with API key from environment variable
# It's crucial to set this environment variable on your server (e.g., Render.com)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize the Generative Model
# Using 'gemini-1.5-flash' as it's a good balance of speed and capability.
# You can change this to 'gemini-1.5-pro' if you need more advanced reasoning,
# but be aware of potential cost implications and rate limits.
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'contents' not in data:
            return jsonify({"error": "Invalid request: 'contents' field is missing."}), 400

        # The 'contents' field is expected to be a list of chat history objects
        chat_history = data['contents']

        # Start a chat session with the model
        # The history should be passed directly to start_chat
        chat_session = model.start_chat(history=chat_history)

        # Get the latest user message from the history
        # Assuming the last item in contents is the current user message
        latest_user_message_parts = chat_history[-1]['parts']

        # Send the latest user message to the chat session
        response = chat_session.send_message(latest_user_message_parts)

        # Extract the text from the response
        ai_response_text = response.text

        return jsonify({"response": ai_response_text})

    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # When running locally, Flask defaults to 127.0.0.1:5000
    # For Render, gunicorn will manage the host/port
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
