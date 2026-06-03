import React, { useState } from 'react';

const ArticleCard = ({ article }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  
  const isFake = article.is_fake;
  const scorePercent = Math.round((article.credibility_score || 0) * 100);
  
  const details = article.score_details || {};

  const toggleDetails = () => {
    setShowDetails(!showDetails);
    if (!showDetails) setShowSummary(false);
  };

  const toggleSummary = () => {
    setShowSummary(!showSummary);
    if (!showSummary) setShowDetails(false);
  };
  
  const keywordArray = article.keywords ? article.keywords.split(',').slice(0, 3) : [];

  return (
    <article className="article-card">
      <div className="article-header">
        <span className="source-badge">{article.source} <span style={{ opacity: 0.5, margin: '0 4px' }}>|</span> {article.category}</span>
        
        <div 
          className={`credibility-badge clickable-badge ${isFake ? 'fake' : 'real'}`}
          onClick={toggleDetails}
          title="Click to see technical logic"
        >
          {isFake ? '⚠️ Flagged Fake' : '✓ Verified'} 
          <span style={{ opacity: 0.7, marginLeft: '4px' }}>{scorePercent}%</span>
        </div>
      </div>
      
      <div className="article-body">
        <h3 className="article-title">{article.title}</h3>
        
        <p className="article-summary">{article.summary || "No intelligence summary available."}</p>

        {showDetails && (
          <div className="score-breakdown">
            <div className="ai-reasoning-box" style={{ background: '#f8fafc', padding: '12px', borderRadius: '6px', borderLeft: `4px solid ${isFake ? 'var(--danger)' : 'var(--success)'}` }}>
              <span style={{ fontWeight: '600', color: '#334155', display: 'block', marginBottom: '6px' }}>AI Reasoning</span>
              <p style={{ color: '#475569', fontSize: '14px', lineHeight: '1.5', margin: 0 }}>
                {details.explanation_text || "No AI reasoning available for this article."}
              </p>
            </div>
            
            {details.fact_check && details.fact_check.fact_check?.claims_found > 0 && (
                <div className="ai-logic-detail" style={{ background: '#fef2f2', border: '1px solid #fee2e2', marginTop: '12px', padding: '12px', borderRadius: '6px' }}>
                    <span className="logic-label" style={{ color: 'var(--danger)', fontWeight: '600' }}>Professional Fact Checks Found:</span>
                    <div style={{ marginTop: '5px' }}>
                        {details.fact_check.fact_check.ratings.map((r, i) => (
                            <span key={i} className="term risk" style={{ background: '#fca5a5', color: '#7f1d1d', padding: '2px 6px', borderRadius: '4px', fontSize: '12px', marginRight: '4px' }}>{r}</span>
                        ))}
                    </div>
                </div>
            )}
          </div>
        )}

        {showSummary && (
          <div className="article-summary-box">
             <div className="summary-scroll">
               {article.full_content.split('\n').map((para, i) => (
                 para.trim() && <p key={i} className="summary-para">{para}</p>
               ))}
             </div>
          </div>
        )}
        
        {keywordArray.length > 0 && !showDetails && !showSummary && (
          <div className="keywords-container" style={{ marginTop: '1rem' }}>
            {keywordArray.map((kw, i) => (
              <span key={i} className="keyword" style={{ background: '#f1f5f9', color: '#64748b' }}>{kw.trim()}</span>
            ))}
          </div>
        )}
      </div>

      <div className="article-footer">
        <div className="article-meta">
          <span>{new Date(article.published_at).toLocaleDateString()}</span>
          {article.author && <span style={{ marginLeft: '8px' }}>• {article.author}</span>}
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="summary-toggle" onClick={toggleSummary}>
            {showSummary ? 'Minify' : 'Preview'}
          </button>
          <a href={article.url} target="_blank" rel="noopener noreferrer" className="read-more">
            Full ↗
          </a>
        </div>
      </div>
    </article>
  );
};

export default ArticleCard;
