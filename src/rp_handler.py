import runpod
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Global variables to store model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the model and tokenizer."""
    global model, tokenizer
    
    print("Loading BGE Reranker model...")
    model_name = "BAAI/bge-reranker-v2-m3"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    model.eval()
    print("Model loaded successfully!")

def compute_score(pairs, normalize=False):
    """Compute relevance scores for query-passage pairs."""
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        
        # Apply sigmoid function to normalize scores between 0 and 1 if requested
        if normalize:
            scores = torch.sigmoid(scores)
        
        return scores.cpu().tolist()

def handler(event):
    """
    This is the handler function that processes incoming requests.
    
    Args:
        event (dict): Contains the input data and request metadata
        
    Returns:
        dict: The result to be returned to the client
    """
    global model, tokenizer
    
    # Load model if not already loaded
    if model is None or tokenizer is None:
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
