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

    # Simple auth check for development stage
    if username == 'demo' and password == 'demo123':
        token = generate_token(username)
        return jsonify({'token': token})

    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/ask', methods=['POST'])
@require_auth
def ask_question():
    """
    Enhanced endpoint for asking questions
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

        result = rag.query(question, use_adaptor=use_adaptor)

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
    Stores source information for better learning
    """
    try:
        data = request.get_json()
        query_id = data.get('query_id')
        rating = data.get('rating')
        comment = data.get('comment', '')

        # Get question and sources from request data or fallback to query_info
        question = data.get('question')
        sources = data.get('sources', [])

        if not query_id or rating is None:
            return jsonify({'error': 'query_id and rating are required'}), 400

        if not isinstance(rating, int) or not 1 <= rating <= 5:
            return jsonify({'error': 'Rating must be an integer between 1 and 5'}), 400

        # If question or sources are missing, retrieve from query_info
        if not question or not sources:
            logger.info(f"Missing question or sources in request, retrieving from query_info for {query_id}")
            query_info = rag.get_query_info(query_id)

            if not query_info:
                logger.warning(f"No query info found for query_id: {query_id}")
                return jsonify({'error': f'Query information not found for query_id: {query_id}'}), 400

            # Use fallback values if not provided in request
            if not question:
                question = query_info.get('question')
                logger.info(f"Retrieved question from query_info: {question[:50]}...")

            if not sources:
                sources = query_info.get('sources_used', [])
                logger.info(f"Retrieved {len(sources)} sources from query_info")

        # Final validation
        if not question:
            return jsonify({'error': 'Question not found in request or query info'}), 400

        print("SOURCES LENGTH FROM APP: ", len(sources))
        logger.info(
            f"Storing feedback for query {query_id}: rating={rating}, question_length={len(question)}, sources_count={len(sources)}")

        # Store feedback with complete information
        feedback_id = rag.store_feedback(query_id, question, rating, comment, sources)

        # Also update the query_info with feedback
        feedback_data = {
            'feedback_id': feedback_id,
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        rag.update_query_feedback(query_id, feedback_data)

        return jsonify({
            'message': 'Feedback stored successfully',
            'trigger_training': len(rag.feedback_data) % 10 == 0,
            'sources_used': len(sources),
            'question_retrieved': question is not None
        })

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

@app.route('/admin/reindex', methods=['POST'])
@require_auth
def reindex_documents():
    """
    Admin endpoint to reindex documents
    Useful when PDF content changes
    """
    try:
        # Clears existing collection
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
            'version': 2.0
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
        }), 503


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('DEBUG', 'False').lower() == 'true')