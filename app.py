from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from rag_system import RAGSystem
from auth import require_auth, generate_token
import logging

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Initialize RAG system
rag = RAGSystem()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/auth/token', methods=['POST'])
def login():
    """Simple token generation for demo purposes"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Simple auth check (in production, use proper authentication)
    if username == 'demo' and password == 'demo123':
        token = generate_token(username)
        return jsonify({'token': token})

    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/ask', methods=['POST'])
@require_auth
def ask_question():
    """Main endpoint for asking questions"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        logger.info(f"Processing question: {question}")

        # Get response from RAG system
        result = rag.query(question)

        return jsonify({
            'answer': result['answer'],
            'confidence': result['confidence'],
            'sources': result['sources'],
            'query_id': result['query_id']
        })

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/feedback', methods=['POST'])
@require_auth
def submit_feedback():
    """Endpoint for submitting feedback"""
    try:
        data = request.get_json()
        query_id = data.get('query_id')
        rating = data.get('rating')
        comment = data.get('comment', '')

        if not query_id or rating is None:
            return jsonify({'error': 'query_id and rating are required'}), 400

        if not isinstance(rating, int) or not 1 <= rating <= 5:
            return jsonify({'error': 'Rating must be an integer between 1 and 5'}), 400

        logger.info(f"Storing feedback for query {query_id}: rating={rating}")

        # Store feedback and update rankings
        rag.store_feedback(query_id, rating, comment)

        return jsonify({'message': 'Feedback stored successfully'})

    except Exception as e:
        logger.error(f"Error storing feedback: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/stats', methods=['GET'])
@require_auth
def get_stats():
    """Get feedback statistics"""
    try:
        stats = rag.get_feedback_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('DEBUG', 'False').lower() == 'true')
