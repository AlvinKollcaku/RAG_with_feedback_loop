# RAG System with Feedback Loop

## Introduction

This project implements an advanced Retrieval-Augmented Generation (RAG) system with an integrated feedback loop mechanism. The system is designed to continuously improve its performance through user feedback and incorporates several sophisticated techniques to enhance retrieval quality and answer generation.

[![RAG System Demo](https://img.youtube.com/vi/F6DEnkGSRow/maxresdefault.jpg)](https://www.youtube.com/watch?v=F6DEnkGSRow) 

*Click the thumbnail above to watch the project demonstration*

## Technologies Used

- **Flask** - Web framework for the API and user interface
- **Chroma** - Vector database for document storage and similarity search
- **OpenAI** - Embeddings generation (OpenAIEmbeddings) and answer generation
  - OpenAI Embeddings for document and query vectorization
  - OpenAI GPT model for response generation

## Dataset

The system utilizes the **FAQ for Python** dataset sourced from Python's official documentation, providing comprehensive question-answer pairs covering various Python topics and use cases.

## Advanced RAG Implementation Features

To enhance the quality and accuracy of the RAG system, several sophisticated techniques have been implemented:

### 1. Query Expansion
- Transforms single user queries into multiple related queries
- Increases the likelihood of retrieving relevant documents
- Improves coverage of potential relevant information

### 2. Embedding Adaptor
- Fine-tunes the embedding model based on user feedback
- Learns from positive and negative user ratings
- Continuously adapts to improve document retrieval relevance

### 3. Reranking with Cross Encoder
- Implements a cross-encoder model for document reranking
- Performs more sophisticated relevance scoring beyond initial vector similarity
- Incorporates historical feedback data to adjust document scores

## Response Rating System

Users can rate each generated response on a scale of **1-5**:
- **5**: Excellent response
- **4**: Good response
- **3**: Average response
- **2**: Poor response
- **1**: Very poor response

## Feedback Loop Mechanism

The system incorporates user ratings into two key components:

### Reranking Enhancement
- Documents that have been frequently included in negatively-rated responses receive lower scores during reranking
- Historical negative feedback reduces the likelihood of retrieving similar problematic documents
- Improves overall system performance over time

### Embedding Adaptor Training
The feedback loop uses user ratings as training labels for the embedding adaptor:

- **Positive Labels (1)**: Responses rated 4 or 5
- **Negative Labels (-1)**: Responses rated 1 or 2
- **Neutral (No contribution)**: Responses rated 3 are excluded from training

This approach ensures that the embedding model learns from clear positive and negative examples while avoiding ambiguous neutral feedback.

## How It Works

1. **Query Processing**: User query is expanded into multiple related queries
2. **Applying the Adaptor**: The embeddings adaptor matrix is applied to each query vector to improve its position in the vector space.
3. **Document Retrieval**: Enhanced embeddings retrieve relevant documents from Chroma
4. **Reranking**: Cross-encoder reranks documents considering historical feedback
5. **Answer Generation**: OpenAI model generates response using top-ranked documents
6. **User Feedback**: User rates the response quality
7. **System Learning**: Feedback updates both reranking scores and embedding adaptor training data (if a specific number of new feedbacks is reached)

## Future Enhancements

- Database integration and deployment

---

*This RAG system demonstrates the power of combining advanced retrieval techniques with continuous learning through user feedback, resulting in a system that improves over time.*
