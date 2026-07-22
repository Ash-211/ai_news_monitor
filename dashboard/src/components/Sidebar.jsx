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
  const isWhatsAppActive = location.pathname === '/whatsapp';
  const isVerifyActive = location.pathname === '/verify';

  return (
    <aside className="sidebar">
      <div className="logo-container" style={{ cursor: 'pointer' }} onClick={() => handleFilterClick('all')}>
        <div className="logo-icon">N</div>
        <div className="brand-name">DailyNewsAI</div>
      </div>
      
      <div className="filter-section">
        <h3>Main Intelligence</h3>
        <button 
          className={`filter-button ${(!isDiscordActive && !isWhatsAppActive && !isVerifyActive && currentFilter === 'all') ? 'active' : ''}`}
          onClick={() => handleFilterClick('all')}
        >
          <span>Global Feed</span>
          <span className="badge">{stats?.total_articles || 0}</span>
        </button>
        <button 
          className={`filter-button ${(!isDiscordActive && !isWhatsAppActive && !isVerifyActive && currentFilter === 'real') ? 'active' : ''}`}
          onClick={() => handleFilterClick('real')}
        >
          <span>Verified Authentic</span>
          <span className="badge">{stats?.real_articles || 0}</span>
        </button>
        <button 
          className={`filter-button ${(!isDiscordActive && !isWhatsAppActive && !isVerifyActive && currentFilter === 'fake') ? 'active' : ''}`}
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
        <button 
          className={`filter-button ${isWhatsAppActive ? 'active' : ''}`}
          onClick={() => navigate('/whatsapp')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            color: isWhatsAppActive ? '#25D366' : 'var(--text-muted)',
            backgroundColor: isWhatsAppActive ? 'rgba(37, 211, 102, 0.1)' : 'transparent',
            fontWeight: isWhatsAppActive ? '600' : '500'
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z" />
          </svg>
          <span>WhatsApp Bot</span>
        </button>
        <button 
          className={`filter-button ${isVerifyActive ? 'active' : ''}`}
          onClick={() => navigate('/verify')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            color: isVerifyActive ? 'var(--accent-primary)' : 'var(--text-muted)',
            backgroundColor: isVerifyActive ? 'rgba(37, 99, 235, 0.08)' : 'transparent',
            fontWeight: isVerifyActive ? '600' : '500'
          }}
        >
          <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <span>Verify URL</span>
        </button>
      </div>

      <div className="filter-section" style={{ marginTop: 'auto' }}>
        <h3>Sector Analysis</h3>
        {stats?.categories && Object.entries(stats.categories).map(([category, count]) => (
          <button 
            key={category}
            className={`filter-button ${(!isDiscordActive && !isWhatsAppActive && !isVerifyActive && currentFilter === category) ? 'active' : ''}`}
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
