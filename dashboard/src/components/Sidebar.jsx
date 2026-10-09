import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Globe, 
  ShieldCheck, 
  Flag, 
  Box, 
  Briefcase, 
  Cpu, 
  Trophy, 
  HelpCircle, 
  Earth, 
  Activity, 
  Landmark, 
  Layers,
  Radar
} from 'lucide-react';

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

  // Helper function to get a specific icon for a sector/category
  const getCategoryIcon = (category) => {
    const cat = category.toLowerCase();
    if (cat === 'business') return <Briefcase size={16} />;
    if (cat === 'sci/tech' || cat === 'technology') return <Cpu size={16} />;
    if (cat === 'sports') return <Trophy size={16} />;
    if (cat === 'world') return <Earth size={16} />;
    if (cat === 'health') return <Activity size={16} />;
    if (cat === 'politics') return <Landmark size={16} />;
    if (cat === 'unknown') return <HelpCircle size={16} />;
    if (cat === 'general') return <Layers size={16} />;
    return <Box size={16} />;
  };

  return (
    <aside className="sidebar">
      <div className="logo-container" style={{ cursor: 'pointer', padding: '0.2rem 0.5rem 1rem 0.5rem' }} onClick={() => handleFilterClick('all')}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '38px',
          height: '38px',
          background: 'rgba(255, 255, 255, 0.05)',
          border: '1px solid var(--border-strong)',
          borderRadius: '10px',
          color: 'var(--text-main)',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)'
        }}>
          <Radar size={20} />
        </div>
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
              {getCategoryIcon(category)} <span style={{ textTransform: 'capitalize' }}>{category}</span>
            </div>
            <span className="badge">{count}</span>
          </button>
        ))}
      </div>
    </aside>
  );
};

export default Sidebar;
