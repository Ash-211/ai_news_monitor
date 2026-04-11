import React from 'react';

const Sidebar = ({ stats, currentFilter, setCurrentFilter }) => {
  return (
    <aside className="sidebar">
      <div className="logo-container">
        <div className="logo-icon">N</div>
        <div className="brand-name">DailyNewsAI</div>
      </div>
      
      <div className="filter-section">
        <h3>Main Intelligence</h3>
        <button 
          className={`filter-button ${currentFilter === 'all' ? 'active' : ''}`}
          onClick={() => setCurrentFilter('all')}
        >
          <span>Global Feed</span>
          <span className="badge">{stats?.total_articles || 0}</span>
        </button>
        <button 
          className={`filter-button ${currentFilter === 'real' ? 'active' : ''}`}
          onClick={() => setCurrentFilter('real')}
        >
          <span>Verified Authentic</span>
          <span className="badge">{stats?.real_articles || 0}</span>
        </button>
        <button 
          className={`filter-button ${currentFilter === 'fake' ? 'active' : ''}`}
          onClick={() => setCurrentFilter('fake')}
        >
          <span>Flagged Fake</span>
          <span className="badge">{stats?.fake_articles || 0}</span>
        </button>
      </div>

      <div className="filter-section" style={{ marginTop: 'auto' }}>
        <h3>Sector Analysis</h3>
        {stats?.categories && Object.entries(stats.categories).map(([category, count]) => (
          <button 
            key={category}
            className={`filter-button ${currentFilter === category ? 'active' : ''}`}
            onClick={() => setCurrentFilter(category)}
          >
            <span style={{ textTransform: 'capitalize' }}>{category}</span>
            <span className="badge">{count}</span>
          </button>
        ))}
      </div>
    </aside>
  );
};

export default Sidebar;
