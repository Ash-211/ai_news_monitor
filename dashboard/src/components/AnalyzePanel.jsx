import React, { useState } from 'react';

const AnalyzePanel = () => {
  const [text, setText] = useState('');
  const [source, setSource] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (text.trim().length < 20) {
      setError('Please enter at least 20 characters of article text.');
      return;
    }
    setError('');
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim(), source: source.trim() || null }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Analysis failed.');
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to the analysis API.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setSource('');
    setResult(null);
    setError('');
  };

  const loadSample = (type) => {
    if (type === 'fake') {
      setText(
        'BREAKING: Secret government documents EXPOSED reveal that world leaders are secretly controlled by an underground organization!!! ' +
        'They DON\'T want you to know this SHOCKING truth about the hidden agenda behind global policies. ' +
        'EXPOSED: Scientists CONFIRM that mainstream media is LYING to the public about everything! Share before they DELETE this!!!'
      );
      setSource('');
    } else {
      setText(
        'The Reserve Bank of India announced a 25 basis point cut in the repo rate on Friday, ' +
        'bringing it down to 6.25 percent, citing stable inflation and robust economic growth. ' +
        'The monetary policy committee voted unanimously in favour of the decision, ' +
        'aiming to support credit growth while maintaining price stability across the economy.'
      );
      setSource('Reuters');
    }
    setResult(null);
    setError('');
  };

  return (
    <div className="analyze-panel">
      {/* Input Section */}
      <div className="analyze-input-section">
        <div className="analyze-input-header">
          <div className="analyze-icon-wrapper">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
              <path d="M11 8v6" />
              <path d="M8 11h6" />
            </svg>
          </div>
          <div>
            <h2 className="analyze-title">Live Article Analyzer</h2>
            <p className="analyze-subtitle">Paste any article text to detect fake news in real-time</p>
          </div>
        </div>

        <div className="sample-buttons">
          <button className="sample-btn fake-sample" onClick={() => loadSample('fake')}>
            ⚠️ Load Fake Sample
          </button>
          <button className="sample-btn real-sample" onClick={() => loadSample('real')}>
            ✓ Load Real Sample
          </button>
        </div>

        <textarea
          className="analyze-textarea"
          placeholder="Paste the full article text here... The more text you provide, the more accurate the analysis will be."
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={8}
        />

        <input
          className="analyze-source-input"
          type="text"
          placeholder="Source name (optional, e.g. BBC, Reuters, Times of India)"
          value={source}
          onChange={(e) => setSource(e.target.value)}
        />

        {error && <div className="analyze-error">{error}</div>}

        <div className="analyze-actions">
          <button className="analyze-btn" onClick={handleAnalyze} disabled={loading}>
            {loading ? (
              <>
                <span className="btn-spinner" /> Analyzing...
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
                  <path d="m9 12 2 2 4-4" />
                </svg>
                Analyze Article
              </>
            )}
          </button>
          <button className="clear-btn" onClick={handleClear} disabled={loading}>Clear</button>
        </div>

        <div className="char-count">{text.length} characters</div>
      </div>

      {/* Results Section */}
      {result && (
        <div className="analyze-results" key={Date.now()}>
          {/* Verdict Banner */}
          <div className={`verdict-banner ${result.is_fake ? 'fake' : 'real'}`}>
            <div className="verdict-icon">
              {result.is_fake ? '⚠️' : '✅'}
            </div>
            <div className="verdict-info">
              <span className="verdict-label">AI Verdict</span>
              <span className="verdict-text">{result.verdict}</span>
            </div>
            <div className="verdict-score">
              <span className="verdict-score-value">{result.credibility_percent}%</span>
              <span className="verdict-score-label">Credibility</span>
            </div>
          </div>

          {/* Credibility Gauge */}
          <div className="result-card">
            <h3 className="result-card-title">Credibility Score</h3>
            <div className="gauge-container">
              <div className="gauge-bg">
                <div
                  className="gauge-fill"
                  style={{
                    width: `${result.credibility_percent}%`,
                    background: result.is_fake
                      ? 'linear-gradient(90deg, #ef4444 0%, #f97316 100%)'
                      : 'linear-gradient(90deg, #10b981 0%, #3b82f6 100%)',
                  }}
                />
              </div>
              <div className="gauge-labels">
                <span>0%</span>
                <span className="gauge-threshold" style={{ left: `${result.threshold * 100}%` }}>
                  ↑ Threshold ({result.threshold * 100}%)
                </span>
                <span>100%</span>
              </div>
            </div>
          </div>

          {/* Score Breakdown */}
          <div className="result-card">
            <h3 className="result-card-title">Score Breakdown</h3>
            <div className="breakdown-list">
              <div className="bd-row">
                <span className="bd-label">🤖 AI Content Analysis (ML)</span>
                <span className="bd-value">{result.score_breakdown.base_ml_percent}%</span>
              </div>
              <div className="bd-row">
                <span className="bd-label">
                  🏢 Source Reputation
                  {result.is_trusted_source && <span className="trusted-tag">TRUSTED</span>}
                </span>
                <span className={`bd-value ${result.score_breakdown.source_bonus > 0 ? 'plus' : ''}`}>
                  {result.score_breakdown.source_bonus > 0 ? `+${result.score_breakdown.source_bonus * 100}%` : '—'}
                </span>
              </div>
              <div className="bd-row">
                <span className="bd-label">🔗 Cross-Reference Corroboration</span>
                <span className={`bd-value ${result.score_breakdown.corroboration_bonus > 0 ? 'plus' : ''}`}>
                  {result.score_breakdown.corroboration_bonus > 0 ? `+${result.score_breakdown.corroboration_bonus * 100}%` : '—'}
                </span>
              </div>
              {result.score_breakdown.isolation_penalty < 0 && (
                <div className="bd-row">
                  <span className="bd-label">🚨 Isolation Penalty (No Corroboration)</span>
                  <span className="bd-value minus">{result.score_breakdown.isolation_penalty * 100}%</span>
                </div>
              )}
            </div>
          </div>

          {/* AI Reasoning */}
          <div className="result-card">
            <h3 className="result-card-title">AI Reasoning</h3>

            <div className="reasoning-grid">
              <div className="reasoning-item">
                <div className="reasoning-metric">
                  <span className="reasoning-label">Sensationalism Index</span>
                  <span className="reasoning-value" style={{ color: result.ai_reasoning.sensationalism_score > 30 ? 'var(--danger)' : 'var(--success)' }}>
                    {result.ai_reasoning.sensationalism_score}%
                  </span>
                </div>
                <div className="mini-bar">
                  <div className="mini-fill risk" style={{ width: `${result.ai_reasoning.sensationalism_score}%` }} />
                </div>
              </div>

              <div className="reasoning-item">
                <div className="reasoning-metric">
                  <span className="reasoning-label">Factual Alignment</span>
                  <span className="reasoning-value" style={{ color: result.ai_reasoning.objectivity_score > 60 ? 'var(--success)' : 'var(--danger)' }}>
                    {result.ai_reasoning.objectivity_score}%
                  </span>
                </div>
                <div className="mini-bar">
                  <div className="mini-fill trust" style={{ width: `${result.ai_reasoning.objectivity_score}%` }} />
                </div>
              </div>
            </div>

            {result.ai_reasoning.caps_ratio > 0 && (
              <div className="extra-stat">
                ALL CAPS ratio: <strong>{(result.ai_reasoning.caps_ratio * 100).toFixed(1)}%</strong>
              </div>
            )}
            {result.ai_reasoning.punctuation_count > 0 && (
              <div className="extra-stat">
                Exclamation/Question marks: <strong>{result.ai_reasoning.punctuation_count}</strong>
              </div>
            )}

            {result.ai_reasoning.trust_keywords?.length > 0 && (
              <div className="keyword-section">
                <span className="kw-heading trust">Trust Influencers</span>
                <div className="kw-tags">
                  {result.ai_reasoning.trust_keywords.map((k, i) => (
                    <span key={i} className="kw-tag trust">{k.word} <small>({k.impact}%)</small></span>
                  ))}
                </div>
              </div>
            )}

            {result.ai_reasoning.risk_keywords?.length > 0 && (
              <div className="keyword-section">
                <span className="kw-heading risk">Risk Influencers</span>
                <div className="kw-tags">
                  {result.ai_reasoning.risk_keywords.map((k, i) => (
                    <span key={i} className="kw-tag risk">{k.word} <small>({k.impact}%)</small></span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Extracted Keywords */}
          {result.keywords?.length > 0 && (
            <div className="result-card">
              <h3 className="result-card-title">Extracted Keywords</h3>
              <div className="kw-tags">
                {result.keywords.map((kw, i) => (
                  <span key={i} className="kw-tag neutral">{kw}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AnalyzePanel;
