import React, { useState, useRef, useCallback } from 'react';
import './DeepfakePage.css';

/**
 * DeepfakePage — Upload an image or video and get an AI-powered
 * deepfake / AI-generation analysis with confidence scores,
 * per-frame timelines (for video), and XAI explanations.
 */
const DeepfakePage = () => {
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  // ── State ──────────────────────────────────────────────────────────
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [mediaType, setMediaType] = useState(null); // 'image' | 'video'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const fileInputRef = useRef(null);

  // ── Drag & Drop handlers ───────────────────────────────────────────
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  // ── File selection ─────────────────────────────────────────────────
  const handleFileSelect = (selectedFile) => {
    setResult(null);
    setError(null);

    const type = selectedFile.type;
    const isImage = type.startsWith('image/');
    const isVideo = type.startsWith('video/');

    if (!isImage && !isVideo) {
      setError('Please upload an image (jpg, png, webp) or video (mp4, avi, mov, webm).');
      return;
    }

    // Size checks
    const maxSize = isVideo ? 50 * 1024 * 1024 : 10 * 1024 * 1024;
    if (selectedFile.size > maxSize) {
      setError(`File too large. Max ${isVideo ? '50' : '10'} MB.`);
      return;
    }

    setFile(selectedFile);
    setMediaType(isImage ? 'image' : 'video');

    // Generate preview URL
    const previewUrl = URL.createObjectURL(selectedFile);
    setPreview(previewUrl);
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  // ── Submit for analysis ────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${API_BASE_URL}/api/deepfake/analyze`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError(`Analysis failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ── Reset ──────────────────────────────────────────────────────────
  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setMediaType(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // ── Credibility Ring (reused pattern from VerifyPage) ──────────────
  const renderVerdictRing = (confidence, isFake) => {
    const pct = Math.round(confidence * 100);
    const radius = 52;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (confidence * circumference);
    const color = isFake
      ? (pct >= 70 ? '#ef4444' : '#f59e0b')   // Red for high-confidence fake, amber for low
      : (pct >= 70 ? '#22c55e' : '#3b82f6');   // Green for high-confidence real, blue for low

    return (
      <div className="df-verdict-ring-container">
        <svg width="120" height="120" viewBox="0 0 120 120">
          {/* Background track */}
          <circle cx="60" cy="60" r={radius} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
          {/* Animated progress arc */}
          <circle
            cx="60" cy="60" r={radius}
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            transform="rotate(-90 60 60)"
            className="df-ring-progress"
          />
        </svg>
        <div className="df-ring-label">
          <span className="df-ring-pct" style={{ color }}>{pct}%</span>
          <span className="df-ring-sublabel">{isFake ? 'Fake' : 'Real'}</span>
        </div>
      </div>
    );
  };

  // ── Frame Timeline (for video results) ─────────────────────────────
  const renderFrameTimeline = (frameResults) => {
    if (!frameResults || frameResults.length === 0) return null;

    const maxConf = Math.max(...frameResults.map(f => f.confidence));

    return (
      <div className="df-timeline">
        <h3 className="df-section-title">
          <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
          </svg>
          Frame-by-Frame Analysis
        </h3>
        <div className="df-timeline-chart">
          {frameResults.map((frame, idx) => {
            const heightPct = (frame.confidence / maxConf) * 100;
            const color = frame.is_fake ? '#ef4444' : '#22c55e';
            return (
              <div key={idx} className="df-timeline-bar-wrapper" title={`Frame ${frame.frame} (${frame.timestamp}s) — ${frame.label}: ${Math.round(frame.confidence * 100)}%`}>
                <div
                  className="df-timeline-bar"
                  style={{
                    height: `${heightPct}%`,
                    backgroundColor: color,
                  }}
                />
                {idx % Math.max(1, Math.floor(frameResults.length / 8)) === 0 && (
                  <span className="df-timeline-label">{frame.timestamp}s</span>
                )}
              </div>
            );
          })}
        </div>
        <div className="df-timeline-legend">
          <span className="df-legend-item"><span className="df-legend-dot" style={{ background: '#22c55e' }} /> Real</span>
          <span className="df-legend-item"><span className="df-legend-dot" style={{ background: '#ef4444' }} /> Fake</span>
        </div>
      </div>
    );
  };

  // ═══════════════════════════════════════════════════════════════════
  //  RENDER
  // ═══════════════════════════════════════════════════════════════════
  return (
    <div className="df-page">
      {/* Header */}
      <div className="df-header">
        <div className="df-header-icon">
          <svg width="28" height="28" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
        </div>
        <div>
          <h1 className="df-title">Deepfake Detector</h1>
          <p className="df-subtitle">Upload an image or video to analyze it for AI-generated or manipulated content</p>
        </div>
      </div>

      {/* Upload Zone */}
      {!file && (
        <div
          className={`df-dropzone ${dragActive ? 'df-dropzone-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp,video/mp4,video/avi,video/quicktime,video/webm"
            onChange={handleInputChange}
            style={{ display: 'none' }}
          />
          <div className="df-dropzone-icon">
            <svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <p className="df-dropzone-text">Drag & drop a file here, or click to browse</p>
          <p className="df-dropzone-hint">Supports JPG, PNG, WEBP images (max 10MB) and MP4, AVI, MOV, WEBM videos (max 50MB)</p>
        </div>
      )}

      {/* Preview + Analyze */}
      {file && !result && !loading && (
        <div className="df-preview-section">
          <div className="df-preview-card">
            {mediaType === 'image' ? (
              <img src={preview} alt="Preview" className="df-preview-media" />
            ) : (
              <video src={preview} controls className="df-preview-media" />
            )}
            <div className="df-preview-info">
              <p className="df-preview-name">{file.name}</p>
              <p className="df-preview-meta">
                {mediaType === 'image' ? '🖼️ Image' : '🎬 Video'} • {(file.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
          </div>
          <div className="df-action-row">
            <button className="df-btn df-btn-secondary" onClick={handleReset}>Cancel</button>
            <button className="df-btn df-btn-primary" onClick={handleAnalyze}>
              <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              Analyze for Deepfake
            </button>
          </div>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && (
        <div className="df-loading">
          <div className="df-loading-card">
            <div className="df-loading-preview skeleton-shimmer" />
            <div className="df-loading-details">
              <div className="skeleton-line skeleton-shimmer" style={{ width: '60%', height: '20px', marginBottom: '12px' }} />
              <div className="skeleton-line skeleton-shimmer" style={{ width: '40%', height: '16px', marginBottom: '24px' }} />
              <div className="skeleton-line skeleton-shimmer" style={{ width: '100%', height: '80px', borderRadius: '12px' }} />
            </div>
          </div>
          <div style={{ textAlign: 'center', marginTop: '20px' }}>
            <p style={{ color: 'var(--text-muted)', fontSize: '1rem', fontWeight: '500' }}>
              {mediaType === 'video' ? 'Extracting & analyzing frames...' : 'Analyzing image...'}
            </p>
            <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: '4px' }}>
              {mediaType === 'video'
                ? 'Frame Extraction → Per-Frame Classification → Score Aggregation'
                : 'Preprocessing → Model Inference → Explanation Generation'}
            </p>
          </div>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="df-error">
          <div className="df-error-icon">❌</div>
          <h3>Analysis Failed</h3>
          <p>{error}</p>
          <button className="df-btn df-btn-secondary" onClick={handleReset} style={{ marginTop: '16px' }}>Try Again</button>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="df-results">
          <div className="df-result-card">
            {/* Verdict Header */}
            <div className="df-result-header">
              {renderVerdictRing(result.confidence, result.is_fake)}
              <div className="df-result-info">
                <h2 className="df-result-verdict" style={{ color: result.is_fake ? '#ef4444' : '#22c55e' }}>
                  {result.is_fake ? '⚠️ Likely Deepfake / AI-Generated' : '✅ Likely Authentic'}
                </h2>
                <div className="df-result-meta">
                  <span className={`df-badge ${result.is_fake ? 'df-badge-fake' : 'df-badge-real'}`}>
                    {result.label}
                  </span>
                  <span className="df-badge df-badge-type">
                    {result.media_type === 'image' ? '🖼️ Image' : '🎬 Video'}
                  </span>
                  {result.filename && (
                    <span className="df-badge df-badge-file">{result.filename}</span>
                  )}
                </div>
              </div>
            </div>

            {/* Media Preview */}
            <div className="df-result-preview">
              {mediaType === 'image' ? (
                <img src={preview} alt="Analyzed" className="df-result-media" />
              ) : (
                <video src={preview} controls className="df-result-media" />
              )}
            </div>

            {/* XAI Explanation */}
            {result.explanation && (
              <div className="df-explanation">
                <h3 className="df-section-title">
                  <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  AI Analysis
                </h3>
                {result.explanation.split('\n').filter(l => l.trim()).map((line, i) => (
                  <p key={i} className="df-explanation-line">{line}</p>
                ))}
              </div>
            )}

            {/* Video-specific: Frame Timeline */}
            {result.media_type === 'video' && result.frame_results && (
              <>
                {renderFrameTimeline(result.frame_results)}
                <div className="df-video-stats">
                  <div className="df-stat">
                    <span className="df-stat-value">{result.total_frames}</span>
                    <span className="df-stat-label">Total Frames</span>
                  </div>
                  <div className="df-stat">
                    <span className="df-stat-value">{result.analyzed_frames}</span>
                    <span className="df-stat-label">Analyzed</span>
                  </div>
                  <div className="df-stat">
                    <span className="df-stat-value">{result.fps}</span>
                    <span className="df-stat-label">FPS</span>
                  </div>
                  <div className="df-stat">
                    <span className="df-stat-value">{result.duration_seconds}s</span>
                    <span className="df-stat-label">Duration</span>
                  </div>
                  <div className="df-stat">
                    <span className="df-stat-value">{Math.round((result.fake_frame_ratio || 0) * 100)}%</span>
                    <span className="df-stat-label">Fake Frames</span>
                  </div>
                </div>
              </>
            )}

            {/* Raw Scores */}
            {result.raw_scores && (
              <div className="df-raw-scores">
                <h3 className="df-section-title">
                  <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Model Confidence Breakdown
                </h3>
                <div className="df-score-bars">
                  {Object.entries(result.raw_scores).map(([label, score]) => (
                    <div key={label} className="df-score-row">
                      <span className="df-score-label">{label}</span>
                      <div className="df-score-bar-track">
                        <div
                          className="df-score-bar-fill"
                          style={{
                            width: `${Math.round(score * 100)}%`,
                            backgroundColor: label.toLowerCase().includes('fake') || label.toLowerCase().includes('ai') ? '#ef4444' : '#22c55e',
                          }}
                        />
                      </div>
                      <span className="df-score-value">{Math.round(score * 100)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Reset Button */}
            <div className="df-action-row" style={{ marginTop: '24px', justifyContent: 'center' }}>
              <button className="df-btn df-btn-secondary" onClick={handleReset}>
                Analyze Another File
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DeepfakePage;
