# Proof of Concept: Scalable and Optimized Graph Implementation

## Overview
This project serves as a **Proof of Concept (PoC)** for a scalable and optimized graph-based implementation written in Python. The primary goal is to validate the algorithmic design and establish a foundation for a future high-performance implementation in **C++**. The system uses **Hierarchical Navigable Small World (HNSW)** principles to build a graph for approximate nearest neighbor (ANN) search, specifically optimized for embedding-based document retrieval tasks.

### Key Features
- **Cosine Similarity Calculation**: Efficiently computes similarity between nodes for ANN tasks.
- **Layered Graph Structure**: Implements multi-layered navigation for hierarchical searches.
- **Heuristic Neighbor Selection**: Optimizes neighbor connections during graph insertion.
- **MongoDB Integration**: Stores graph data, enabling persistence and retrieval.
- **Text Embedding**: Uses pre-trained BERT from the Hugging Face Transformers library for feature extraction.

### Current Limitations
- This implementation prioritizes algorithm correctness and clarity over performance.
- The optimization and scalability will be addressed in the upcoming C++ version.

## Technologies Used
- **Programming Languages**: Python
- **Libraries/Frameworks**:
  - [NumPy](https://numpy.org/) for numerical operations.
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/) for text embeddings.
  - [pandas](https://pandas.pydata.org/) for data manipulation.
  - [heapq](https://docs.python.org/3/library/heapq.html) for efficient heap operations.
  - [MongoDB](https://www.mongodb.com/) for graph persistence.
- **Skills Applied and Gained**:
  - Advanced algorithm design for ANN.
  - Efficient use of Python data structures (heaps, deques).
  - MongoDB schema design and integration.
  - Text processing and embedding techniques using NLP models.
  - Performance profiling and optimization.

## Project Structure
- **Core Classes**:
  - `Node`: Represents a graph node with attributes such as score, document, and neighbor relationships.
  - `Rough_Draft_HSNW`: Implements the HNSW-based graph structure.
- **Functions**:
  - `create_node`: Generates nodes with embeddings.
  - `insert`: Inserts nodes into the graph and updates neighbor connections.
  - `search`: Performs ANN search over the graph.
  - `search_layer` and `select_neighbors_heuristic`: Optimize node selection for hierarchical searches.
- **Utilities**:
  - `cosine_similarity`: Computes similarity scores.
  - MongoDB operations to persist and retrieve graph nodes.

## Usage Instructions
### Prerequisites
1. Install the required Python libraries:
   ```bash
   pip install numpy pandas transformers pymongo
   ```
2. Ensure MongoDB is running and configured for the project.

### Running the Project
1. **Graph Creation**:
   - Load a dataset (e.g., an Excel file) containing the textual data.
   - Use the `create_graph` function to initialize the graph.
2. **Querying the Graph**:
   - Use the `search_graph` function to interactively query the graph with text inputs.

### Example
```python
# Initialize the graph for the 'technology' topic
graph = Rough_Draft_HSNW("technology", M=5, Mmax=5, Mmxa0=10)

# Create the graph
create_graph()

# Query the graph
search_graph()
```

## Future Plans
- **C++ Implementation**: Transition to a high-performance version in C++.
- **Batch Optimization**: Implement bulk updates for MongoDB operations.
- **Concurrency**: Add support for parallel processing.
- **Scalability**: Optimize memory and computation for large datasets.

## Conclusion
This project demonstrates the feasibility of using HNSW algorithm in building a search index. By transitioning to C++, I aim to achieve significant performance gains and scalability, making it suitable for large-scale applications.

