#!/usr/bin/env python3
"""Simple API server for performance testing"""

from fastapi import FastAPI
from pydantic import BaseModel
import re
import hashlib
import time
import uvicorn

app = FastAPI()

# Pre-compiled patterns
PATTERNS = {
    'SSN': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'EMAIL': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'PHONE': re.compile(r'\b(?:\+?1[\s-]?)?\(?[0-9]{3}\)?[\s-]?[0-9]{3}[\s-]?[0-9]{4}\b'),
}

# Cache
cache = {}

class DetectRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/v2/detect")
def detect(request: DetectRequest):
    start = time.perf_counter()

    # Check cache
    text_hash = hashlib.md5(request.text.encode()).hexdigest()
    if text_hash in cache:
        result = cache[text_hash]
        processing_time = (time.perf_counter() - start) * 1000
        return {
            "detected": result,
            "processing_time_ms": processing_time,
            "cached": True
        }

    # Process
    result = {}
    for name, pattern in PATTERNS.items():
        matches = pattern.findall(request.text)
        if matches:
            result[name] = matches

    # Cache result
    cache[text_hash] = result

    processing_time = (time.perf_counter() - start) * 1000

    return {
        "detected": result,
        "processing_time_ms": processing_time,
        "cached": False
    }

if __name__ == "__main__":
    print("Starting test API server on http://localhost:8888")
    uvicorn.run(app, host="0.0.0.0", port=8888)