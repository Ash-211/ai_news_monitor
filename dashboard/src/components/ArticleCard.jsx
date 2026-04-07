import React from 'react';

const ArticleCard = ({ article }) => {
  const isFake = article.is_fake;
  // Credibility score is a percentage (0 to 1) 
  // Wait, credibility score in my fake_news.py is 1 = Real, 0 = Fake because 0 is Authentic, credibility = prob of class 0.
  const scorePercent = Math.round((article.credibility_score || 0) * 100);
  
  // Clean up keywords safely
  const keywordArray = article.keywords ? article.keywords.split(',').slice(0, 3) : [];

  return (
    <article className="article-card">
      <div className="article-header">
        <span className="source-badge">{article.source} • {article.category}</span>
        
        <div className={`credibility-badge ${isFake ? 'fake' : 'real'}`}>
          {isFake ? '⚠️ Flagged Fake' : '✓ Verified'} 
          <span style={{ opacity: 0.8 }}>({scorePercent}%)</span>
        </div>
      </div>
      
      {/* Score progress bar */}
      <div className="article-header" style={{ paddingTop: '8px', paddingBottom: '0' }}>
         <div className="score-bar" style={{ width: '100%' }}>
            <div 
              className="score-fill" 
              style={{ 
                width: `${scorePercent}%`, 
                backgroundColor: isFake ? 'var(--danger)' : 'var(--success)'
              }} 
            />
         </div>
      </div>

      <div className="article-body">
        <h3 className="article-title">{article.title}</h3>
        <p className="article-summary">{article.summary || "No summary available."}</p>
        
        {keywordArray.length > 0 && (
          <div className="keywords-container">
            {keywordArray.map((kw, i) => (
              <span key={i} className="keyword">{kw.trim()}</span>
            ))}
          </div>
        )}
      </div>

      <div className="article-footer">
        <div className="article-meta">
          <span>{new Date(article.published_at).toLocaleDateString()}</span>
          {article.author && <span>By {article.author}</span>}
        </div>
        <a href={article.url} target="_blank" rel="noopener noreferrer" className="read-more">
          Read Full ↗
        </a>
      </div>
    </article>
  );
};

export default ArticleCard;
