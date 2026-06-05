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

      <div className="filter-section">
        <h3>Integrations</h3>
        <button 
          className={`filter-button ${isDiscordActive ? 'active' : ''}`}
          onClick={() => navigate('/discord')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            color: isDiscordActive ? '#5865F2' : 'var(--text-muted)',
            backgroundColor: isDiscordActive ? 'rgba(88, 101, 242, 0.1)' : 'transparent',
            fontWeight: isDiscordActive ? '600' : '500'
          }}
        >
          <svg width="18" height="18" viewBox="0 0 127.14 96.36" fill="currentColor">
            <path d="M107.7,8.07A105.15,105.15,0,0,0,77.26,0a77.19,77.19,0,0,0-3.3,6.83A96.67,96.67,0,0,0,53.22,6.83,77.19,77.19,0,0,0,49.88,0,105.15,105.15,0,0,0,19.44,8.07C3.66,31.58-1.95,54.65,1,77.53A105.73,105.73,0,0,0,32,96.36a77.7,77.7,0,0,0,6.63-10.85,68.43,68.43,0,0,1-10.5-5c.89-.65,1.76-1.34,2.58-2.06a75.22,75.22,0,0,0,73.1,0c.82.72,1.69,1.41,2.58,2.06a68.43,68.43,0,0,1-10.5,5,77.7,77.7,0,0,0,6.63,10.85,105.73,105.73,0,0,0,31.06-18.83C129.89,48.37,123.6,25.43,107.7,8.07ZM42.45,65.69C36.18,65.69,31,60,31,53S36.18,40.36,42.45,40.36,53.83,46,53.83,53,48.72,65.69,42.45,65.69Zm42.24,0C78.41,65.69,73.24,60,73.24,53S78.41,40.36,84.69,40.36,96.07,46,96.07,53,91,65.69,84.69,65.69Z" />
          </svg>
          <span>Discord Bot</span>
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
