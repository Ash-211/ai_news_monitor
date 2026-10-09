import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Globe, ShieldCheck, Flag, Box } from 'lucide-react';

const Sidebar = ({ stats, currentFilter, setCurrentFilter }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleFilterClick = (filter) => {
    setCurrentFilter(filter);
    if (location.pathname !== '/') {
      navigate('/');
    }
  };

  const isHome = location.pathname === '/';

  return (
    <aside className="sidebar">
      <div className="logo-container" style={{ cursor: 'pointer' }} onClick={() => handleFilterClick('all')}>
        <div className="logo-icon">N</div>
        <div className="brand-name">DailyNewsAI</div>
      </div>
      
      <div className="filter-section">
        <h3>Main Intelligence</h3>
        <button 
          className={`filter-button ${(isHome && currentFilter === 'all') ? 'active' : ''}`}
          onClick={() => handleFilterClick('all')}
        >
          <div className="filter-button-left">
            <Globe size={16} /> <span>Global Feed</span>
          </div>
          <span className="badge">{stats?.total_articles || 0}</span>
        </button>
        <button 
          className={`filter-button ${(isHome && currentFilter === 'real') ? 'active' : ''}`}
          onClick={() => handleFilterClick('real')}
        >
          <div className="filter-button-left">
            <ShieldCheck size={16} color="var(--success)" /> <span>Verified Authentic</span>
          </div>
          <span className="badge">{stats?.real_articles || 0}</span>
        </button>
        <button 
          className={`filter-button ${(isHome && currentFilter === 'fake') ? 'active' : ''}`}
          onClick={() => handleFilterClick('fake')}
        >
          <div className="filter-button-left">
            <Flag size={16} color="var(--danger)" /> <span>Flagged Fake</span>
          </div>
          <span className="badge">{stats?.fake_articles || 0}</span>
        </button>
      </div>

      <div className="filter-section" style={{ marginTop: '1rem' }}>
        <h3>Sector Analysis</h3>
        {stats?.categories && Object.entries(stats.categories).map(([category, count]) => (
          <button 
            key={category}
            className={`filter-button ${(isHome && currentFilter === category) ? 'active' : ''}`}
            onClick={() => handleFilterClick(category)}
          >
            <div className="filter-button-left">
              <Box size={16} /> <span style={{ textTransform: 'capitalize' }}>{category}</span>
            </div>
            <span className="badge">{count}</span>
          </button>
        ))}
      </div>
    </aside>
  );
};

export default Sidebar;
