import React, { useState } from 'react';

const ArticleCard = ({ article }) => {
  const [showDetails, setShowDetails] = useState(false);
  
  const isFake = article.is_fake;
  const scorePercent = Math.round((article.credibility_score || 0) * 100);
  
  // Extra details from API
  const details = article.score_details || {};
  
  // Clean up keywords safely
  const keywordArray = article.keywords ? article.keywords.split(',').slice(0, 3) : [];

  return (
    <article className="article-card">
      <div className="article-header">
        <span className="source-badge">{article.source} • {article.category}</span>
        
        <div 
          className={`credibility-badge clickable-badge ${isFake ? 'fake' : 'real'}`}
          onClick={() => setShowDetails(!showDetails)}
          title="Click to see score breakdown"
        >
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
        
        {showDetails && (
          <div className="score-breakdown">
            <div className="breakdown-item">
              <span className="breakdown-label">AI Content Analysis</span>
              <span className="breakdown-value">{Math.round(details.base_ml * 100)}%</span>
            </div>
            
            {/* Improved AI Logic Section with Percentages */}
            {details.ai_logic && (
              <div className="ai-logic-detail">
                <div className="logic-row">
                  <span className="logic-label">Sensationalism Index:</span>
                  <span className="logic-value">{details.ai_logic.sensationalism_score}%</span>
                </div>
                <div className="logic-bar">
                  <div 
                    className="logic-fill risk" 
                    style={{ width: `${details.ai_logic.sensationalism_score}%` }} 
                  />
                </div>

                <div className="logic-row" style={{ marginTop: '10px' }}>
                  <span className="logic-label">Factual Alignment:</span>
                  <span className="logic-value">{details.ai_logic.objectivity_score}%</span>
                </div>
                <div className="logic-bar">
                  <div 
                    className="logic-fill trust" 
                    style={{ width: `${details.ai_logic.objectivity_score}%` }} 
                  />
                </div>
                
                {details.ai_logic.trust_keywords?.length > 0 && (
                  <div className="logic-row keywords" style={{ marginTop: '12px' }}>
                    <span className="logic-label">Trust Influencers:</span>
                    <div className="logic-terms">
                      {details.ai_logic.trust_keywords.map((item, i) => (
                        <span key={i} className="term trust">
                          {item.word} <small>({item.impact}%)</small>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                {details.ai_logic.risk_keywords?.length > 0 && (
                  <div className="logic-row keywords">
                    <span className="logic-label">Risk Influencers:</span>
                    <div className="logic-terms">
                      {details.ai_logic.risk_keywords.map((item, i) => (
                        <span key={i} className="term risk">
                          {item.word} <small>({item.impact}%)</small>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            
            <div style={{ margin: '10px 0', borderTop: '1px solid rgba(255,255,255,0.05)' }} />

            {details.is_trusted && (
              <div className="breakdown-item">
                <span className="breakdown-label">Authoritative Source</span>
                <span className="breakdown-value plus">+15%</span>
              </div>
            )}
            
            {details.corroboration_count >= 1 && (
              <div className="breakdown-item">
                <span className="breakdown-label">Cross-Reference Boost ({details.corroboration_count})</span>
                <span className="breakdown-value plus">+10%</span>
              </div>
            )}
            
            {!details.is_trusted && details.corroboration_count === 0 && (
              <div className="breakdown-item">
                <span className="breakdown-label">Isolation Penalty</span>
                <span className="breakdown-value minus">-15%</span>
              </div>
            )}
          </div>
        )}

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
