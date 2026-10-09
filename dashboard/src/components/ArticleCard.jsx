import React, { useState } from 'react';
import { ExternalLink, Maximize2, Minimize2, AlertTriangle, CheckCircle, ChevronDown, ChevronUp } from 'lucide-react';

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
        <span className="category-tag">{article.source} <span style={{ opacity: 0.5, margin: '0 4px' }}>|</span> {article.category}</span>
        
        <div 
          className={`verdict-tag ${isFake ? 'fake' : 'real'}`}
          onClick={toggleDetails}
          title="Click to see technical logic"
          style={{ cursor: 'pointer', transition: 'var(--transition)' }}
          onMouseOver={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
          onMouseOut={(e) => e.currentTarget.style.transform = 'scale(1)'}
        >
          {isFake ? <AlertTriangle size={14} /> : <CheckCircle size={14} />}
          {isFake ? 'Flagged Fake' : 'Verified'} 
          <span style={{ opacity: 0.8, marginLeft: '4px' }}>{scorePercent}%</span>
        </div>
      </div>
      
      <div className="article-content">
        <h3 className="article-title">
          <a href={article.url} target="_blank" rel="noopener noreferrer">
            {article.title}
          </a>
        </h3>
        
        <p className="article-summary">{article.summary || "No intelligence summary available."}</p>

        {showDetails && (
          <div className="fact-check-box" style={{ borderLeft: `4px solid ${isFake ? 'var(--danger)' : 'var(--success)'}` }}>
            <h4>AI Reasoning</h4>
            <p>
              {details.explanation_text || "No AI reasoning available for this article."}
            </p>
            
            {details.fact_check && details.fact_check.fact_check?.claims_found > 0 && (
                <div style={{ background: 'rgba(255, 69, 58, 0.1)', border: '1px solid rgba(255, 69, 58, 0.2)', marginTop: '12px', padding: '12px', borderRadius: '8px' }}>
                    <span style={{ color: 'var(--danger)', fontWeight: '600', fontSize: '0.8rem', textTransform: 'uppercase', display: 'block', marginBottom: '8px' }}>Professional Fact Checks Found:</span>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {details.fact_check.fact_check.ratings.map((r, i) => (
                            <span key={i} style={{ background: 'var(--danger)', color: '#fff', padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem', fontWeight: '600' }}>{r}</span>
                        ))}
                    </div>
                </div>
            )}
          </div>
        )}

        {showSummary && (
          <div className="fact-check-box" style={{ maxHeight: '350px', overflowY: 'auto' }}>
             <div>
               {article.full_content && article.full_content.split('\n').map((para, i) => (
                 para.trim() && <p key={i} style={{ marginBottom: '0.8rem' }}>{para}</p>
               ))}
               
               {article.image_url && article.image_status !== 'discarded' && (
                 <div style={{ marginTop: '16px', textAlign: 'center' }}>
                   <img 
                     src={article.image_url} 
                     alt="Article preview" 
                     style={{ 
                       maxWidth: '100%', 
                       maxHeight: '300px', 
                       borderRadius: '8px', 
                       border: '1px solid var(--border-light)',
                       boxShadow: 'var(--shadow-sm)'
                     }} 
                   />
                   {article.image_status === 'deepfake' && (
                     <div style={{ color: 'var(--danger)', fontSize: '0.75rem', fontWeight: '600', marginTop: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                       <AlertTriangle size={14} /> Warning: Highly likely to be a Deepfake/AI-generated.
                     </div>
                   )}
                 </div>
               )}
             </div>
          </div>
        )}
        
        {keywordArray.length > 0 && !showDetails && !showSummary && (
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '1rem' }}>
            {keywordArray.map((kw, i) => (
              <span key={i} style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-light)', color: 'var(--text-muted)', padding: '2px 8px', borderRadius: '6px', fontSize: '0.7rem', fontWeight: '500' }}>{kw.trim()}</span>
            ))}
          </div>
        )}
      </div>

      <div className="article-footer">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>{new Date(article.published_at).toLocaleDateString()}</span>
          {article.author && <span>• {article.author}</span>}
        </div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <button 
            onClick={toggleSummary}
            style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.8rem', fontWeight: '500' }}
            onMouseOver={(e) => e.currentTarget.style.color = 'var(--text-main)'}
            onMouseOut={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            {showSummary ? <><Minimize2 size={14} /> Minify</> : <><Maximize2 size={14} /> Preview</>}
          </button>
          <a href={article.url} target="_blank" rel="noopener noreferrer" style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-main)', textDecoration: 'none', fontSize: '0.8rem', fontWeight: '600' }}>
            Full <ExternalLink size={14} />
          </a>
        </div>
      </div>
    </article>
  );
};

export default ArticleCard;
