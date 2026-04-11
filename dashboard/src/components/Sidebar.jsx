import React from 'react';

const Sidebar = ({ stats, currentFilter, setCurrentFilter }) => {
  return (
    <aside className="sidebar">
      <div className="logo-container">
        <div className="logo-icon">AI</div>
        <div className="brand-name">NewsSense</div>
      </div>
      
      <div className="filter-section">
        <h3>News Feed</h3>
        <button 
          className={`filter-button ${currentFilter === 'all' ? 'active' : ''}`}
          onClick={() => setCurrentFilter('all')}
        >
          <span>All Articles</span>
          <span className="badge">{stats?.total_articles || 0}</span>
        </button>
        <button 
          className={`filter-button ${currentFilter === 'real' ? 'active' : ''}`}
          onClick={() => setCurrentFilter('real')}
        >
          <span>Verified Real</span>
          <span className="badge" style={{ color: 'var(--success)' }}>{stats?.real_articles || 0}</span>
        </button>
        <button 
          className={`filter-button ${currentFilter === 'fake' ? 'active' : ''}`}
          onClick={() => setCurrentFilter('fake')}
        >
          <span>Flagged Fake</span>
          <span className="badge" style={{ color: 'var(--danger)' }}>{stats?.fake_articles || 0}</span>
        </button>
      </div>

      {/* Live Analyzer Button */}
      <div className="filter-section">
        <h3>Tools</h3>
        <button 
          className={`filter-button analyze-nav-btn ${currentFilter === 'analyze' ? 'active' : ''}`}
          onClick={() => setCurrentFilter('analyze')}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
            Analyze Article
          </span>
          <span className="badge" style={{ 
            background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
            color: 'white',
            fontWeight: '700',
            fontSize: '0.65rem',
            letterSpacing: '0.05em'
          }}>LIVE</span>
        </button>
      </div>

      <div className="filter-section" style={{ marginTop: 'auto' }}>
        <h3>Topics</h3>
        {stats?.categories && Object.entries(stats.categories).map(([category, count]) => (
          <button 
            key={category}
            className={`filter-button ${currentFilter === category ? 'active' : ''}`}
            onClick={() => setCurrentFilter(category)}
          >
            <span>{category}</span>
            <span className="badge">{count}</span>
          </button>
        ))}
      </div>
    </aside>
  );
};

export default Sidebar;
