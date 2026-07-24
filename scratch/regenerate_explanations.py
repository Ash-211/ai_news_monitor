import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.database import get_session, Article
from src.intelligence.fake_news import generate_explanation

def run():
    session = get_session()
    articles = session.query(Article).all()
    print(f"Found {len(articles)} articles to update.")
    
    updated = 0
    for a in articles:
        if not a.score_details:
            continue
            
        try:
            details = json.loads(a.score_details)
        except:
            continue
            
        score = a.credibility_score
        if score is None:
            continue
            
        trust_factors = details.get('trust_factors', [])
        risk_factors = details.get('risk_factors', [])
        
        # Don't regenerate if it already has a long explanation (greater than 200 chars)
        if details.get('explanation') and len(details['explanation']) > 200:
            continue
            
        try:
            print(f"Generating for ID {a.id}...")
            new_exp = generate_explanation(score, trust_factors, risk_factors, a.title, a.source)
            details['explanation'] = new_exp
            a.score_details = json.dumps(details)
            updated += 1
            print(f"Updated ID {a.id}")
            
            # Commit every 5 articles
            if updated % 5 == 0:
                session.commit()
        except Exception as e:
            print(f"Failed ID {a.id}: {e}")
            
    session.commit()
    session.close()
    print(f"Successfully updated {updated} articles.")

if __name__ == "__main__":
    run()
