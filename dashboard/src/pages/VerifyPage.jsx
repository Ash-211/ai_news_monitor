import React, { useState } from 'react';
import './VerifyPage.css';

const VerifyPage = () => {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  const handleVerify = async () => {
    const trimmed = url.trim();
    if (!trimmed) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: trimmed }),
      });

      const data = await res.json();

      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError('Network error — could not reach the server. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !loading) handleVerify();
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setUrl('');
  };

  // Credibility ring colour
  const getScoreColor = (score) => {
    if (score === null || score === undefined) return '#94a3b8';
    if (score >= 0.6) return 'var(--success)';
    if (score >= 0.4) return 'var(--warning)';
    return 'var(--danger)';
  };

  const renderCredibilityRing = (score) => {
    const pct = score != null ? Math.round(score * 100) : 0;
    const color = getScoreColor(score);
    const angle = score != null ? score * 360 : 0;

    return (
      <div className="credibility-ring-container">
        <div
          className="credibility-ring"
          style={{
            background: `conic-gradient(${color} ${angle}deg, #e2e8f0 ${angle}deg)`,
          }}
        >
          <div className="credibility-ring-inner">
            <span className="credibility-ring-value" style={{ color }}>
              {score != null ? `${pct}%` : 'N/A'}
            </span>
          </div>
        </div>
        <span className="credibility-ring-label">Credibility</span>
      </div>
    );
  };

  const renderVerdictBadge = (isFake) => {
    if (isFake === null || isFake === undefined) {
      return <span className="verify-meta-badge verdict-unknown">⬜ Unknown</span>;
    }
    return isFake
      ? <span className="verify-meta-badge verdict-fake">🚩 Potentially Misleading</span>
      : <span className="verify-meta-badge verdict-real">✅ Authentic</span>;
  };

  return (
    <div className="verify-page">
      {/* Hero */}
      <div className="verify-hero">
        <div className="verify-hero-icon">
          <svg width="32" height="32" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
        </div>
        <h1>Verify Any Article</h1>
        <p>
          Paste a news article URL below and our AI pipeline will scrape, classify,
          and fact-check it in real time — powered by DistilBERT, TF-IDF, and external verification APIs.
        </p>
      </div>

      {/* Input Bar */}
      <div className="verify-input-container">
        <input
          type="url"
          className="verify-url-input"
          placeholder="https://example.com/news/article..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button
          className="verify-submit-btn"
          onClick={handleVerify}
          disabled={loading || !url.trim()}
        >
          <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          Verify
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div className="verify-results">
          <div className="verify-result-card" style={{ opacity: 0.8 }}>
            {/* Header Skeleton */}
            <div className="verify-result-header">
              <div className="credibility-ring-container">
                <div className="skeleton-shimmer" style={{ width: '110px', height: '110px', borderRadius: '50%' }} />
              </div>
              <div className="verify-result-info" style={{ width: '100%' }}>
                <div className="skeleton-line skeleton-shimmer" style={{ width: '70%', height: '24px', marginBottom: '16px' }} />
                <div className="verify-result-meta">
                  <div className="skeleton-line skeleton-shimmer" style={{ width: '120px', height: '28px', borderRadius: '8px' }} />
                  <div className="skeleton-line skeleton-shimmer" style={{ width: '100px', height: '28px', borderRadius: '8px' }} />
                </div>
              </div>
            </div>

            {/* Body Skeleton */}
            <div className="verify-result-body">
              <div className="verify-section">
                <div className="skeleton-line skeleton-shimmer" style={{ width: '150px', height: '14px', marginBottom: '12px' }} />
                <div className="skeleton-line skeleton-shimmer" style={{ width: '100%', height: '16px', marginBottom: '8px' }} />
                <div className="skeleton-line skeleton-shimmer" style={{ width: '90%', height: '16px', marginBottom: '8px' }} />
                <div className="skeleton-line skeleton-shimmer" style={{ width: '75%', height: '16px' }} />
              </div>

              <div className="verify-section">
                <div className="skeleton-line skeleton-shimmer" style={{ width: '120px', height: '14px', marginBottom: '12px' }} />
                <div style={{ display: 'flex', gap: '8px' }}>
                  <div className="skeleton-line skeleton-shimmer" style={{ width: '80px', height: '28px', borderRadius: '999px' }} />
                  <div className="skeleton-line skeleton-shimmer" style={{ width: '110px', height: '28px', borderRadius: '999px' }} />
                  <div className="skeleton-line skeleton-shimmer" style={{ width: '90px', height: '28px', borderRadius: '999px' }} />
                </div>
              </div>
            </div>

            {/* Footer Skeleton */}
            <div className="verify-result-footer" style={{ borderTop: 'none', background: '#fafafa' }}>
              <div className="skeleton-line skeleton-shimmer" style={{ width: '30%', height: '14px' }} />
            </div>
          </div>
          
          <div style={{ textAlign: 'center', marginTop: '20px' }}>
             <p style={{ color: 'var(--text-muted)', fontSize: '1rem', fontWeight: '500' }}>Analyzing article...</p>
             <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '4px' }}>Scraping → Cleaning → Classifying → Detecting → Fact-checking</p>
          </div>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="verify-error">
          <div className="verify-error-icon">❌</div>
          <h3>Verification Failed</h3>
          <p>{error}</p>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="verify-results">
          <div className="verify-result-card">
            {/* Header */}
            <div className="verify-result-header">
              {renderCredibilityRing(result.credibility_score)}

              <div className="verify-result-info">
                <h2 className="verify-result-title">
                  <a href={result.url} target="_blank" rel="noopener noreferrer">
                    {result.title || 'Untitled Article'}
                  </a>
                </h2>

                <div className="verify-result-meta">
                  {renderVerdictBadge(result.is_fake)}

                  {result.category && (
                    <span className="verify-meta-badge category">
                      📂 {result.category}
                      {result.category_confidence != null && (
                        <span style={{ opacity: 0.7, marginLeft: 4 }}>
                          {Math.round(result.category_confidence * 100)}%
                        </span>
                      )}
                    </span>
                  )}

                  {result.source && (
                    <span className="verify-meta-badge source">
                      🌐 {result.source}
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* Body */}
            <div className="verify-result-body">
              {/* AI Explanation */}
              {result.explanation && (
                <div className="verify-section" style={{
                  borderLeft: `4px solid ${getScoreColor(result.credibility_score)}`
                }}>
                  <div className="verify-section-title">
                    🤖 AI Analysis
                  </div>
                  <p>{result.explanation}</p>
                </div>
              )}

              {/* Keywords */}
              {result.keywords && result.keywords.length > 0 && (
                <div className="verify-section">
                  <div className="verify-section-title">
                    🔑 Extracted Keywords
                  </div>
                  <div className="verify-keywords">
                    {result.keywords.map((kw, i) => (
                      <span key={i} className="verify-keyword-pill">{kw}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Fact-Check */}
              {result.fact_check && (
                <div className="verify-section">
                  <div className="verify-section-title">
                    🔎 External Fact-Check
                  </div>

                  <div className="verify-factcheck-grid">
                    <div className="verify-factcheck-item">
                      <span className="fc-value">
                        {Math.round((result.fact_check.verification_score || 0) * 100)}%
                      </span>
                      <span className="fc-label">Verification Score</span>
                    </div>

                    {result.fact_check.cross_reference && (
                      <div className="verify-factcheck-item">
                        <span className="fc-value">
                          {result.fact_check.cross_reference.total_results || 0}
                        </span>
                        <span className="fc-label">Other Outlets</span>
                      </div>
                    )}

                    {result.fact_check.fact_check && result.fact_check.fact_check.claims_found > 0 && (
                      <div className="verify-factcheck-item">
                        <span className="fc-value">
                          {result.fact_check.fact_check.claims_found}
                        </span>
                        <span className="fc-label">Fact-Check Claims</span>
                      </div>
                    )}
                  </div>

                  {/* Matching sources */}
                  {result.fact_check.cross_reference?.matching_sources?.length > 0 && (
                    <div className="verify-factcheck-sources">
                      {result.fact_check.cross_reference.matching_sources.map((src, i) => (
                        <span key={i} className="verify-factcheck-source-tag">{src}</span>
                      ))}
                    </div>
                  )}

                  {/* Google ratings */}
                  {result.fact_check.fact_check?.ratings?.length > 0 && (
                    <div className="verify-factcheck-sources" style={{ marginTop: '0.5rem' }}>
                      {result.fact_check.fact_check.ratings.map((r, i) => (
                        <span key={i} className="verify-factcheck-source-tag" style={{
                          background: '#fef2f2',
                          color: 'var(--danger)',
                          border: '1px solid #fecaca'
                        }}>
                          {r}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="verify-result-footer">
              <button className="verify-again-btn" onClick={handleReset}>
                ← Verify Another
              </button>
              <a
                href={result.url}
                target="_blank"
                rel="noopener noreferrer"
                className="verify-read-full"
              >
                Read Full Article ↗
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VerifyPage;
