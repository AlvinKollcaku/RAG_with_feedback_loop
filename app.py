from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from rag_system import EnhancedRAGSystem
from auth import require_auth, generate_token
import logging
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Initialize enhanced RAG system
# Using environment variables for production configuration
rag = EnhancedRAGSystem(
    collection_name=os.getenv('COLLECTION_NAME', 'faq_documents'),
    persist_directory=os.getenv('CHROMA_PERSIST_DIR', './chroma_db'),
    pdf_path=os.getenv('PDF_PATH', 'data/faqs.pdf')
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Background task for periodic model updates
def background_training():
    """Periodically retrain embedding adaptor based on feedback"""
    import time
    while True:
        time.sleep(3600)  # Wait 1 hour
        try:
            logger.info("Starting background embedding adaptor training...")
            rag.train_embedding_adaptor()
            logger.info("Background training completed")
        except Exception as e:
            logger.error(f"Background training failed: {str(e)}")


# Start background thread
if os.getenv('ENABLE_BACKGROUND_TRAINING', 'true').lower() == 'true':
    training_thread = threading.Thread(target=background_training, daemon=True)
    training_thread.start()


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
    """
    Enhanced endpoint for asking questions
    Now includes:
    - Query expansion details in response
    - Cross-encoder scores for transparency
    - Option to disable embedding adaptor
    """
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        use_adaptor = data.get('use_adaptor', True)  # Allow disabling adaptor

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        logger.info(f"Processing question: {question}")

        # Get response from enhanced RAG system
        result = rag.query(question, use_adaptor=use_adaptor)

        # Add request metadata
        result['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'use_adaptor': use_adaptor,
            'model_version': '2.0'
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/feedback', methods=['POST'])
@require_auth
def submit_feedback():
    """
    Enhanced feedback endpoint
    Now stores source information for better learning
    """
    try:
        data = request.get_json()
        query_id = data.get('query_id')
        rating = data.get('rating')
        comment = data.get('comment', '')
        sources = data.get('sources', [])  # Include which sources were helpful

        if not query_id or rating is None:
            return jsonify({'error': 'query_id and rating are required'}), 400

        if not isinstance(rating, int) or not 1 <= rating <= 5:
            return jsonify({'error': 'Rating must be an integer between 1 and 5'}), 400

        logger.info(f"Storing feedback for query {query_id}: rating={rating}")

        # Store feedback with source information
        rag.store_feedback(query_id, rating, comment, sources)

        return jsonify({
            'message': 'Feedback stored successfully',
            'trigger_training': len(rag.feedback_data) % 50 == 0
        })

    except Exception as e:
        logger.error(f"Error storing feedback: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/stats', methods=['GET'])
@require_auth
def get_stats():
    """Get enhanced statistics including model performance"""
    try:
        stats = rag.get_feedback_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/admin/train', methods=['POST'])
@require_auth
def trigger_training():
    """
    Admin endpoint to manually trigger embedding adaptor training
    Useful for testing or after significant feedback collection
    """
    try:
        num_epochs = request.get_json().get('num_epochs', 50)

        # Run training in background
        def train():
            rag.train_embedding_adaptor(num_epochs=num_epochs)

        thread = threading.Thread(target=train)
        thread.start()

        return jsonify({
            'message': 'Training started in background',
            'num_epochs': num_epochs
        })
    except Exception as e:
        logger.error(f"Error triggering training: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/admin/reindex', methods=['POST'])
@require_auth
def reindex_documents():
    """
    Admin endpoint to reindex documents
    Useful when PDF content changes
    """
    try:
        # Clear existing collection
        rag.chroma_client.delete_collection(rag.collection_name)
        rag._initialize_collection()
        rag._load_and_index_documents()

        return jsonify({
            'message': 'Documents reindexed successfully',
            'document_count': rag.collection.count()
        })
    except Exception as e:
        logger.error(f"Error reindexing: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with system status"""
    try:
        status = {
            'status': 'healthy',
            'components': {
                'chroma': 'connected',
                'document_count': rag.collection.count(),
                'feedback_count': len(rag.feedback_data),
                'adaptor_trained': os.path.exists(rag.adaptor_path)
            },
            'version': '2.0'
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('DEBUG', 'False').lower() == 'true')