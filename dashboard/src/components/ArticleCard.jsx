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
            <div className="breakdown-item">
              <span className="breakdown-label">ML Content Analysis</span>
              <span className="breakdown-value">{Math.round(details.base_ml * 100)}%</span>
            </div>
            
            {details.ai_logic && (
              <div className="ai-logic-detail">
                <div className="logic-row">
                  <span className="logic-label">Sensationalism:</span>
                  <span className="logic-value" style={{ color: details.ai_logic.sensationalism_score > 50 ? 'var(--danger)' : 'var(--text-muted)' }}>
                    {details.ai_logic.sensationalism_score}%
                  </span>
                </div>
                <div className="logic-bar">
                  <div className="logic-fill risk" style={{ width: `${details.ai_logic.sensationalism_score}%` }} />
                </div>

                <div className="logic-row">
                  <span className="logic-label">Objectivity:</span>
                  <span className="logic-value" style={{ color: 'var(--success)' }}>{details.ai_logic.objectivity_score}%</span>
                </div>
                <div className="logic-bar">
                  <div className="logic-fill trust" style={{ width: `${details.ai_logic.objectivity_score}%` }} />
                </div>
              </div>
            )}

            {details.verification_boost !== 0 && (
              <div className="breakdown-item">
                <span className="breakdown-label">External Verification</span>
                <span className={`breakdown-value ${details.verification_boost > 0 ? 'plus' : 'minus'}`}>
                  {details.verification_boost > 0 ? '+' : ''}{Math.round(details.verification_boost * 100)}%
                </span>
              </div>
            )}

            {details.is_trusted && (
              <div className="breakdown-item">
                <span className="breakdown-label">Authoritative Source</span>
                <span className="breakdown-value plus">+15%</span>
              </div>
            )}
            
            {(details.corroboration_count > 0 || details.bonus_applied > 0) && (
              <div className="breakdown-item">
                <span className="breakdown-label">Cross-Reference Bonus</span>
                <span className="breakdown-value plus">Enabled</span>
              </div>
            )}

            {details.fact_check && details.fact_check.fact_check?.claims_found > 0 && (
                <div className="ai-logic-detail" style={{ background: '#fef2f2', border: '1px solid #fee2e2', marginTop: '12px' }}>
                    <span className="logic-label" style={{ color: 'var(--danger)' }}>Professional Fact Checks Found:</span>
                    <div style={{ marginTop: '5px' }}>
                        {details.fact_check.fact_check.ratings.map((r, i) => (
                            <span key={i} className="term risk">{r}</span>
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
