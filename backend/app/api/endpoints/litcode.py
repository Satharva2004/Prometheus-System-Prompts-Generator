import re
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.api.endpoints.ai import get_groq_client

router = APIRouter()

system_prompt = """You are a code complexity analyzer. Think step by step before answering.

Step 1 - Identify all loops and their bounds (nested = multiplicative, sequential = dominant term)
Step 2 - Identify recursive calls and derive recurrence relation if present
Step 3 - Identify library calls and their known complexity (e.g. sort = O(n log n), toString = O(log n))
Step 4 - Identify extra space used (variables, strings, arrays, recursion stack)
Step 5 - Output ONLY a JSON object, no markdown, no backticks, nothing else

Output this exact structure:
{
  "time": {
    "complexity": "O(...)",
    "reason": "one line explanation"
  },
  "space": {
    "complexity": "O(...)",
    "reason": "one line explanation"
  },
  "confidence": "High | Medium | Low",
  "note": "any edge case, assumption, or bounded input worth flagging — or null"
}

Confidence rules:
- High: straightforward loops, no recursion, no custom data structures
- Medium: recursion with clear recurrence, or library calls involved
- Low: amortized structures, mutual recursion, or unknown data structure internals"""

# --- Data Models ---
class ComplexityAnalysisRequest(BaseModel):
    code: str

# --- Endpoints ---
@router.post("/analyze-complexity")
async def analyze_complexity(request: ComplexityAnalysisRequest):
    client = get_groq_client()
    if not client.api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Analyze this code:\n\n{request.code}",
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stop=None
        )

        raw = chat_completion.choices[0].message.content
        if not raw:
            raise ValueError("Empty response from AI")

        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)
            cleaned = cleaned.strip()

        result = json.loads(cleaned)
        return {"success": True, "data": result}

    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or "rate limit" in err_msg:
            raise HTTPException(status_code=429, detail="Too many requests, try again in a moment")
        
        if isinstance(e, json.JSONDecodeError):
            raise HTTPException(status_code=500, detail="Model returned malformed JSON")
            
        raise HTTPException(status_code=500, detail=str(e))
