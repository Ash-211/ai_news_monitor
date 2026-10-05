import os
import urllib.parse
import feedparser
import google.generativeai as genai
from dotenv import load_dotenv

# Load old DistilBERT logic
from src.intelligence.fake_news import detect_fake_news, load_fake_news_detector

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def fetch_evidence(query):
    """Searches Google News for the claim and returns the top 3 headlines."""
    print(f"\n[RAG] Searching Google News for: '{query}'...")
    encoded_query = urllib.parse.quote_plus(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(rss_url)
    evidence = []
    for entry in feed.entries[:3]:
        source = entry.get('source', {}).get('title', 'Unknown Source')
        title = entry.get('title', '')
        evidence.append(f"- {source}: {title}")
    
    if not evidence:
        return "No major news outlets are reporting this."
    return "\n".join(evidence)

def rag_evaluate(claim, evidence):
    """Uses Gemini to compare the claim against the fetched evidence."""
    prompt = f"""
    You are an elite journalistic fact-checker. 
    You must evaluate a suspect claim by comparing it to evidence fetched from trusted news sources.
    
    SUSPECT CLAIM: {claim}
    
    EVIDENCE FROM TRUSTED SOURCES:
    {evidence}
    
    Task: Does the suspect claim heavily contradict the evidence, or is it completely absent from the news cycle? 
    Output ONLY a JSON in this format, with no markdown formatting:
    {{"credibility_score": 0.1, "reasoning": "brief explanation"}}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:-3].strip()
        if text.startswith("```"): text = text[3:-3].strip()
        import json
        return json.loads(text)
    except Exception as e:
        return {"credibility_score": 0.5, "reasoning": f"Error: {e}"}

def run_comparison(claim):
    print("="*60)
    print(f"[CLAIM]: {claim}")
    print("="*60)
    
    # --- 1. Old Method (DistilBERT Only) ---
    print("\n1. Old Method (DistilBERT Analysis):")
    model, tokenizer = load_fake_news_detector()
    is_fake, distil_score, breakdown = detect_fake_news(title=claim, content="", model=model, tokenizer=tokenizer)
    print(f"   => Credibility Score: {int(distil_score * 100)}%")
    print(f"   => Verdict: {'Fake' if is_fake else 'Real'}")
    
    # --- 2. New Method (RAG + LLM) ---
    print("\n2. New Method (RAG Fact-Checking):")
    evidence = fetch_evidence(claim)
    print(f"   [Fetched Evidence]:\n{evidence}")
    
    rag_result = rag_evaluate(claim, evidence)
    rag_score = float(rag_result.get("credibility_score", 0.5))
    print(f"\n   => Credibility Score: {int(rag_score * 100)}%")
    print(f"   => Reasoning: {rag_result.get('reasoning')}")
    print("="*60)

if __name__ == "__main__":
    # Test 1: A gramatically perfect but completely fake claim
    fake_claim = "Supreme Court of India officially bans the use of WhatsApp starting next month due to privacy concerns."
    run_comparison(fake_claim)
    
    # Test 2: A real, breaking news claim
    real_claim = "Ratan Tata, former chairman of Tata Group, admitted to Breach Candy hospital in Mumbai."
    run_comparison(real_claim)
