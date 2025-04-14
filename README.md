# BGE Reranker v2 M3 RunPod Serverless

This repository contains the code to deploy the BAAI/bge-reranker-v2-m3 model as a serverless endpoint on RunPod.

## Overview

The BGE Reranker v2 M3 is a multilingual reranker model that takes a query and a set of passages as input and returns relevance scores. This serverless deployment allows you to use the model through a simple API.

## Local Testing

To test the handler locally:

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the handler:
   ```
   cd src
   python rp_handler.py
   ```

## Building and Deploying

### 1. Build the Docker Image

Replace `[YOUR_USERNAME]` with your Docker Hub username:

```bash
docker build --platform linux/amd64 -t [YOUR_USERNAME]/bge-reranker-v2-m3:latest .
```

### 2. Push the Image to Docker Hub

```bash
docker push [YOUR_USERNAME]/bge-reranker-v2-m3:latest
```

### 3. Deploy on RunPod

1. Go to the [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Select "Docker Image" under "Custom Source"
4. Enter your Docker image URL: `docker.io/[YOUR_USERNAME]/bge-reranker-v2-m3:latest`
5. Name your endpoint
6. Under "Worker Configuration", select GPU type (at least 16GB recommended)
7. Click "Create Endpoint"

## API Usage

Once deployed, you can use the endpoint with the following JSON structure:

```json
{
  "input": {
    "query": "Your query here",
    "passages": [
      "Passage 1 text",
      "Passage 2 text",
      "Passage 3 text"
    ],
    "normalize": true
  }
}
```

The response will be in the following format:

```json
{
  "success": true,
  "results": [
    {
      "passage": "Most relevant passage",
      "score": 0.95
    },
    {
      "passage": "Second most relevant passage",
      "score": 0.75
    },
    {
      "passage": "Least relevant passage",
      "score": 0.25
    }
  ]
}
```

The results are sorted by score in descending order.

## Parameters

- `query`: The search query (required)
- `passages`: An array of text passages to rank (required)
- `normalize`: Whether to normalize scores to [0,1] range using sigmoid function (optional, default: false)
