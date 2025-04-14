import runpod
from FlagEmbedding import FlagReranker

# Global variable to store reranker
reranker = None

def load_model():
    """Load the reranker model."""
    global reranker
    
    print("Loading BGE Reranker model...")
    model_name = "BAAI/bge-reranker-v2-m3"
    
    # Initialize the reranker with FP16 for faster computation
    reranker = FlagReranker(model_name, use_fp16=True)
    
    print("Model loaded successfully!")

def compute_score(pairs, normalize=False):
    """Compute relevance scores for query-passage pairs."""
    # FlagReranker handles GPU usage internally
    scores = reranker.compute_score(pairs, normalize=normalize)
    
    return scores

def handler(event):
    """
    This is the handler function that processes incoming requests.
    
    Args:
        event (dict): Contains the input data and request metadata
        
    Returns:
        dict: The result to be returned to the client
    """
    global reranker
    
    # Load model if not already loaded
    if reranker is None:
        load_model()
    
    # Extract input data
    input_data = event.get("input", {})
    
    # Check if we have the required inputs
    if not input_data:
        return {"error": "No input provided"}
    
    # Get query and passages
    query = input_data.get("query")
    passages = input_data.get("passages", [])
    normalize = input_data.get("normalize", False)
    
    # Validate inputs
    if not query:
        return {"error": "No query provided"}
    if not passages:
        return {"error": "No passages provided"}
    
    # Prepare pairs for scoring
    pairs = [[query, passage] for passage in passages]
    
    # Compute scores
    try:
        scores = compute_score(pairs, normalize)
        
        # Prepare results
        results = []
        for i, score in enumerate(scores):
            results.append({
                "passage": passages[i],
                "score": score
            })
        
        # Sort results by score in descending order
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "success": True,
            "results": results
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Start the serverless function
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
