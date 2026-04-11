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
