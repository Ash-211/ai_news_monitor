import React, { useState, useEffect } from 'react';
import './TrendingPage.css';

const PLATFORM_DEFAULTS = {
  reddit:   { name: "Reddit",       icon: "🟠", risk: "high",     color: "#FF4500" },
  twitter:  { name: "X / Twitter",  icon: "𝕏",  risk: "critical", color: "#000000" },
  facebook: { name: "Facebook",     icon: "📘", risk: "critical", color: "#1877F2" },
};

const TrendingPage = () => {
  const [selectedPlatforms, setSelectedPlatforms] = useState(
    new Set(["reddit", "twitter", "facebook"])
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(() => {
    const saved = sessionStorage.getItem('trendingScanResult');
    return saved ? JSON.parse(saved) : null;
  });
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState("all"); // all | fake | real
  const [expandedItems, setExpandedItems] = useState(new Set());

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

  useEffect(() => {
    if (result) {
      sessionStorage.setItem('trendingScanResult', JSON.stringify(result));
    } else {
      sessionStorage.removeItem('trendingScanResult');
    }
  }, [result]);

  useEffect(() => {
    const saved = sessionStorage.getItem('trendingScanResult');
    if (!saved) {
      // Fetch latest background scan on mount
      fetch(`${API_BASE_URL}/api/trending/latest`)
        .then(res => res.json())
        .then(data => {
          if (data && !data.error && data.items && data.items.length > 0) {
            setResult(data);
          }
        })
        .catch(err => console.error("Could not fetch latest trending:", err));
    }
  }, [API_BASE_URL]);

  const togglePlatform = (key) => {
    setSelectedPlatforms((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        if (next.size > 1) next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const toggleExpand = (hash) => {
    setExpandedItems((prev) => {
      const next = new Set(prev);
      if (next.has(hash)) next.delete(hash);
      else next.add(hash);
      return next;
    });
  };

  const handleScan = async () => {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/trending/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          platforms: Array.from(selectedPlatforms),
          reddit_limit: 5,
        }),
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

  const getScoreLabel = (score) => {
    if (score === null || score === undefined) return '?';
    return `${Math.round(score * 100)}%`;
  };

  const getScoreClass = (analysis) => {
    if (!analysis || analysis.is_fake === null) return 'unknown';
    return analysis.is_fake ? 'fake' : 'real';
  };

  const getFilteredItems = () => {
    if (!result?.items) return [];
    
    // Completely drop gnews items
    const validItems = result.items.filter(i => PLATFORM_DEFAULTS[i.platform]);
    
    if (filter === 'all') return validItems;
    if (filter === 'fake') return validItems.filter(i => i.analysis?.is_fake === true);
    if (filter === 'real') return result.items.filter(i => i.analysis?.is_fake === false);
    // platform filter
    return result.items.filter(i => i.platform === filter && PLATFORM_DEFAULTS[i.platform]);
  };

  const formatTime = (iso) => {
    if (!iso) return '';
    try {
      return new Date(iso).toLocaleString('en-IN', {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit'
      });
    } catch { return iso; }
  };

  return (
    <div className="trending-page">
      {/* Hero */}
      <div className="trending-hero">
        <div className="trending-hero-icon">
          <svg width="36" height="36" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
              d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <h1>Social Misinformation Radar</h1>
        <p>
          Scan trending content across social media platforms where misinformation
          spreads the most powered by AI-driven fake news detection.
        </p>
      </div>

      {/* Platform Selector */}
      {!loading && (
        <>
          <div className="trending-platforms">
            {Object.entries(PLATFORM_DEFAULTS).map(([key, plat]) => (
              <button
                key={key}
                className={`platform-chip ${selectedPlatforms.has(key) ? 'selected' : ''}`}
                onClick={() => togglePlatform(key)}
              >
                <span className="platform-chip-icon">{plat.icon}</span>
                <span>{plat.name}</span>
                <span className={`platform-risk ${plat.risk}`}>{plat.risk}</span>
              </button>
            ))}
          </div>

          <div className="trending-scan-bar">
            <button
              className="trending-scan-btn"
              onClick={handleScan}
              disabled={loading || selectedPlatforms.size === 0}
            >
              <svg width="22" height="22" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Scan {selectedPlatforms.size} Platform{selectedPlatforms.size !== 1 ? 's' : ''} for Fake News
            </button>
          </div>
        </>
      )}

      {/* Loading */}
      {loading && (
        <div className="trending-loading">
          <div className="trending-radar">
            <div className="trending-radar-ring" />
            <div className="trending-radar-ring" />
            <div className="trending-radar-ring" />
            <div className="trending-radar-sweep" />
            <div className="trending-radar-dot" />
            <div className="trending-radar-dot" />
            <div className="trending-radar-dot" />
          </div>
          <h3>Scanning Social Media...</h3>
          <p>Scraping trending content and running AI analysis</p>
          <div className="trending-loading-steps">
            <div className="trending-loading-step active">
              <div className="step-dot" />
              <span>Scraping platforms</span>
            </div>
            <div className="trending-loading-step">
              <div className="step-dot" />
              <span>Deduplicating</span>
            </div>
            <div className="trending-loading-step">
              <div className="step-dot" />
              <span>AI Detection</span>
            </div>
            <div className="trending-loading-step">
              <div className="step-dot" />
              <span>Generating verdicts</span>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="trending-error">
          <h3>❌ Scan Failed</h3>
          <p>{error}</p>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="trending-results">
          {/* Stats Cards */}
          <div className="trending-stats">
            <div className="trending-stat-card accent">
              <span className="trending-stat-value">{result.stats?.total_scanned || 0}</span>
              <span className="trending-stat-label">Posts Analyzed</span>
            </div>
            <div className="trending-stat-card">
              <span className="trending-stat-value" style={{ color: '#94a3b8' }}>
                {result.stats?.non_news_skipped || 0}
              </span>
              <span className="trending-stat-label">Noise Skipped</span>
            </div>
            <div className="trending-stat-card danger">
              <span className="trending-stat-value">{result.stats?.flagged_fake || 0}</span>
              <span className="trending-stat-label">Flagged Fake</span>
            </div>
            <div className="trending-stat-card success">
              <span className="trending-stat-value">{result.stats?.authentic || 0}</span>
              <span className="trending-stat-label">Likely Authentic</span>
            </div>
          </div>

          {/* Platform Breakdown */}
          {result.stats?.platforms && (
            <div className="trending-platform-breakdown">
              {Object.entries(result.stats.platforms).filter(([platform]) => PLATFORM_DEFAULTS[platform]).map(([platform, count]) => {
                const plat = PLATFORM_DEFAULTS[platform] || {};
                return (
                  <div className="platform-breakdown-chip" key={platform}>
                    <span>{plat.icon || '📌'}</span>
                    <span>{plat.name || platform}</span>
                    <span className="platform-breakdown-count">{count}</span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Filter Tabs */}
          <div className="trending-filter-tabs">
            <button
              className={`trending-filter-tab ${filter === 'all' ? 'active' : ''}`}
              onClick={() => setFilter('all')}
            >
              All ({result.items?.length || 0})
            </button>
            <button
              className={`trending-filter-tab ${filter === 'fake' ? 'active' : ''}`}
              onClick={() => setFilter('fake')}
            >
              🚩 Flagged ({result.stats?.flagged_fake || 0})
            </button>
            <button
              className={`trending-filter-tab ${filter === 'real' ? 'active' : ''}`}
              onClick={() => setFilter('real')}
            >
              ✅ Authentic ({result.stats?.authentic || 0})
            </button>
            {Object.entries(result.stats?.platforms || {}).filter(([platform]) => PLATFORM_DEFAULTS[platform]).map(([platform]) => {
              const plat = PLATFORM_DEFAULTS[platform] || {};
              const count = result.items?.filter(i => i.platform === platform).length || 0;
              return (
                <button
                  key={platform}
                  className={`trending-filter-tab ${filter === platform ? 'active' : ''}`}
                  onClick={() => setFilter(platform)}
                >
                  {plat.icon} {plat.name || platform} ({count})
                </button>
              );
            })}
          </div>

          {/* Results Header */}
          <div className="trending-results-header">
            <h2>
              {filter === 'all' && 'All Scanned Content'}
              {filter === 'fake' && '🚩 Flagged Misinformation'}
              {filter === 'real' && '✅ Likely Authentic Content'}
              {!['all', 'fake', 'real'].includes(filter) && `${PLATFORM_DEFAULTS[filter]?.icon || ''} ${PLATFORM_DEFAULTS[filter]?.name || filter} Results`}
            </h2>
            <div className="trending-results-actions" style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
              <span className="trending-results-time">
                Scanned: {formatTime(result.scanned_at)}
              </span>
              <button 
                className="fetch-latest-btn" 
                onClick={handleScan}
                style={{
                  background: 'var(--brand-primary, #6366f1)',
                  color: 'white',
                  border: 'none',
                  padding: '8px 16px',
                  borderRadius: '6px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}
                onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-1px)'}
                onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
              >
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
                Fetch Latest
              </button>
            </div>
          </div>

          {/* Items List */}
          <div className="trending-items-list">
            {getFilteredItems().length === 0 ? (
              <div className="trending-empty">
                <div className="trending-empty-icon">🔍</div>
                <h3>No items match this filter</h3>
                <p>Try selecting a different filter above.</p>
              </div>
            ) : (
              getFilteredItems().map((item, idx) => {
                const analysis = item.analysis || {};
                const isExpanded = expandedItems.has(item.hash);
                const scoreClass = getScoreClass(analysis);
                const plat = PLATFORM_DEFAULTS[item.platform] || {};

                return (
                  <div
                    key={item.hash || idx}
                    className={`trending-item ${analysis.is_fake ? 'flagged' : 'authentic'}`}
                  >
                    <div className="trending-item-header">
                      <div className={`trending-item-score ${scoreClass}`}>
                        <div className="score-percentage">{getScoreLabel(analysis.credibility_score)}</div>
                        <div style={{ fontSize: '0.35em', opacity: 0.8, marginTop: '2px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Credibility</div>
                      </div>
                      <div className="trending-item-info">
                        <div className="trending-item-title">
                          <a href={item.url} target="_blank" rel="noopener noreferrer">
                            {item.title}
                          </a>
                        </div>
                        <div className="trending-item-meta">
                          <span className={`trending-item-badge platform-${item.platform}`}>
                            {plat.icon} {plat.name || item.platform}
                          </span>
                          <span className={`trending-item-badge verdict-${scoreClass}`}>
                            {analysis.is_fake ? '🚩 Flagged' : '✅ Authentic'}
                          </span>
                          {item.source && (
                            <span className="trending-item-badge source">
                              {item.source}
                            </span>
                          )}
                          {item.engagement?.upvotes > 0 && (
                            <span className="trending-item-engagement">
                              ▲ {item.engagement.upvotes.toLocaleString()}
                            </span>
                          )}
                          {item.engagement?.comments > 0 && (
                            <span className="trending-item-engagement">
                              💬 {item.engagement.comments.toLocaleString()}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Expandable AI Explanation */}
                    <div className="trending-item-body">
                      <button
                        className="trending-item-explanation-toggle"
                        onClick={() => toggleExpand(item.hash)}
                      >
                        {isExpanded ? '▾ Hide' : '▸ Show'} AI Analysis
                      </button>
                      {isExpanded && analysis.explanation && (
                        <div className="trending-item-explanation">
                          {analysis.explanation}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Scan Again */}
          <div className="trending-scan-again">
            <button
              className="trending-scan-again-btn"
              onClick={() => { setResult(null); setError(null); setFilter('all'); }}
            >
              ← Scan Again
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrendingPage;
