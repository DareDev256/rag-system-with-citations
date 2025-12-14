import os
import openai
from typing import List, Dict, Any
from src.llm.prompt import RAG_PROMPT_TEMPLATE, CLASSIFICATION_PROMPT_TEMPLATE, build_context_str
from src.utils.timing import measure_latency

# Simple cache for demo purposes to avoid duplicate API calls if restarting
_llm_client = None

def get_llm_client():
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback for demo if key not present, or raise error
            print("Warning: OPENAI_API_KEY not found in env.")
        _llm_client = openai.OpenAI(api_key=api_key)
    return _llm_client

@measure_latency
def classify_query(query: str) -> str:
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # or gpt-4o-mini
            messages=[
                {"role": "system", "content": "You are a precise classifier."},
                {"role": "user", "content": CLASSIFICATION_PROMPT_TEMPLATE.format(query=query)}
            ],
            temperature=0,
            max_tokens=10
        )
        category = response.choices[0].message.content.strip().lower()
        if category not in ["factual", "exploratory", "ambiguous"]:
            return "exploratory" # Default
        return category
    except Exception as e:
        print(f"Classification error: {e}")
        return "factual"

@measure_latency
def synthesize_answer(query: str, search_results: List[Dict]) -> Dict[str, Any]:
    client = get_llm_client()
    
    context_str = build_context_str(search_results)
    prompt = RAG_PROMPT_TEMPLATE.format(context_str=context_str, query=query)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a grounded QA assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # Low temp for grounding
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        
        # In a real system, we might parse citations from the answer string to structure them
        # For now, we return the text and the raw citations used
        return {
            "answer": answer,
            "citations_used": search_results, # Returning all retrieved for now, ideally filter
            "confidence": 1.0 # Placeholder
        }
    except Exception as e:
        print(f"Synthesis error: {e}")
        return {
            "answer": "Error generating answer.",
            "citations_used": [],
            "confidence": 0.0
        }
