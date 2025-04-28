from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model (moved outside endpoints for better performance)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)

# Conversation storage with thread lock for safety
conversation_states = {}
lock = threading.Lock()

def generate_response(user_input, conversation_id="default"):
    """Generate response using DialoGPT model"""
    try:
        # Encode the user input
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            response_ids = model.generate(
                input_ids,
                max_length=1000,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and clean the response
        response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response
    
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error processing your request."

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint for the API"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id', 'default')
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
        
        with lock:
            # Initialize conversation if new
            if conversation_id not in conversation_states:
                conversation_states[conversation_id] = {
                    "history": [],
                    "past_user_inputs": [],
                    "generated_responses": []
                }
            
            # Generate response
            bot_response = generate_response(user_message)
            
            # Update conversation state
            conversation_states[conversation_id]["past_user_inputs"].append(user_message)
            conversation_states[conversation_id]["generated_responses"].append(bot_response)
            conversation_states[conversation_id]["history"].append({
                "user": user_message,
                "bot": bot_response
            })
            
            return jsonify({
                "response": bot_response,
                "conversation_id": conversation_id,
                "history": conversation_states[conversation_id]["history"]
            })
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation history"""
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id', 'default')
        
        with lock:
            if conversation_id in conversation_states:
                del conversation_states[conversation_id]
            
            return jsonify({
                "status": "success",
                "message": "Conversation reset",
                "conversation_id": conversation_id
            })
    
    except Exception as e:
        print(f"Error resetting conversation: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/status', methods=['GET'])
def status_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "model": "microsoft/DialoGPT-medium",
        "device": device
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)