import os
import torch
from transformers import AutoTokenizer, AutoModel
from sqlalchemy import text
from src.ingestion.database import get_session, Article, SocialMediaPost

# We use a tiny but powerful model for generating text embeddings (384 dimensions)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_local_embed_tokenizer = None
_local_embed_model = None

def get_embedding(text_content: str) -> list[float]:
    """Generates a 384-dimensional vector embedding locally (free, zero API credits)."""
    global _local_embed_tokenizer, _local_embed_model
    if not text_content or not text_content.strip():
        return []
        
    try:
        if _local_embed_tokenizer is None or _local_embed_model is None:
            _local_embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
            _local_embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL)
            _local_embed_model.eval()

        inputs = _local_embed_tokenizer(
            text_content[:1000], padding=True, truncation=True, max_length=512, return_tensors='pt'
        )
        with torch.no_grad():
            outputs = _local_embed_model(**inputs)
            # Mean pooling
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            return vector
    except Exception as e:
        print(f"[Event Linker] Local embedding generation failed: {e}")
        return []

def link_social_post_to_news(post_id: int, similarity_threshold: float = 0.7) -> list[dict]:
    """
    Finds verified news articles that talk about the exact same event 
    as the verified social media post using Vector Cosine Similarity.
    """
    session = get_session()
    
    # 1. Get the social media post
    post = session.query(SocialMediaPost).filter(SocialMediaPost.id == post_id).first()
    if not post or post.is_fake:
        session.close()
        return []
        
    # 2. Ensure it has a vector embedding
    if post.embedding is None:
        text_to_embed = f"{post.title} {post.clean_content or post.raw_content}"
        vector = get_embedding(text_to_embed)
        if not vector:
            session.close()
            return []
        post.embedding = vector
        session.commit()
    
    # 3. Perform Vector Similarity Search against Verified News Articles!
    # <= operator in pgvector means Cosine Distance. 
    # Similarity = 1 - Distance. So Distance < (1 - threshold).
    distance_threshold = 1.0 - similarity_threshold
    
    # We query the DB for the closest news articles where is_fake == False
    matched_articles = session.query(Article).filter(
        Article.is_fake == False,
        Article.embedding != None,
        Article.embedding.cosine_distance(post.embedding) < distance_threshold
    ).order_by(
        Article.embedding.cosine_distance(post.embedding)
    ).limit(3).all()
    
    results = [{"id": a.id, "title": a.title, "source": a.source, "url": a.url} for a in matched_articles]
    session.close()
    return results

def get_related_social_posts_for_news(article_id: int, similarity_threshold: float = 0.7) -> list[dict]:
    """Reverse search: Get verified social media posts related to a verified news article."""
    session = get_session()
    
    article = session.query(Article).filter(Article.id == article_id).first()
    if not article or article.is_fake:
        session.close()
        return []
        
    if article.embedding is None:
        text_to_embed = f"{article.title} {article.clean_content or article.raw_content}"
        vector = get_embedding(text_to_embed)
        if not vector:
            session.close()
            return []
        article.embedding = vector
        session.commit()
        
    distance_threshold = 1.0 - similarity_threshold
    
    matched_posts = session.query(SocialMediaPost).filter(
        SocialMediaPost.is_fake == False,
        SocialMediaPost.embedding != None,
        SocialMediaPost.embedding.cosine_distance(article.embedding) < distance_threshold
    ).order_by(
        SocialMediaPost.embedding.cosine_distance(article.embedding)
    ).limit(3).all()
    
    results = [{"id": p.id, "title": p.title, "source": p.source, "url": p.url} for p in matched_posts]
    session.close()
    return results
