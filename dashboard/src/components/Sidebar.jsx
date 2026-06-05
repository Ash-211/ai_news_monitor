import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

const Sidebar = ({ stats, currentFilter, setCurrentFilter }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleFilterClick = (filter) => {
    setCurrentFilter(filter);
    if (location.pathname !== '/') {
      navigate('/');
    }
  };

  const isDiscordActive = location.pathname === '/discord';

  return (
    <aside className="sidebar">
      <div className="logo-container" style={{ cursor: 'pointer' }} onClick={() => handleFilterClick('all')}>
        <div className="logo-icon">N</div>
        <div className="brand-name">DailyNewsAI</div>
      </div>
      
      <div className="filter-section">
        <h3>Main Intelligence</h3>
        <button 
          className={`filter-button ${(!isDiscordActive && currentFilter === 'all') ? 'active' : ''}`}
          onClick={() => handleFilterClick('all')}
        >
          <span>Global Feed</span>
          <span className="badge">{stats?.total_articles || 0}</span>
        </button>
        <button 
          className={`filter-button ${(!isDiscordActive && currentFilter === 'real') ? 'active' : ''}`}
          onClick={() => handleFilterClick('real')}
        >
          <span>Verified Authentic</span>
          <span className="badge">{stats?.real_articles || 0}</span>
        </button>
        <button 
          className={`filter-button ${(!isDiscordActive && currentFilter === 'fake') ? 'active' : ''}`}
          onClick={() => handleFilterClick('fake')}
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
            className={`filter-button ${(!isDiscordActive && currentFilter === category) ? 'active' : ''}`}
            onClick={() => handleFilterClick(category)}
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
