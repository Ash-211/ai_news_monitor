import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Search } from 'lucide-react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import ArticleCard from './components/ArticleCard';
import DiscordLanding from './pages/DiscordLanding';
import WhatsAppLanding from './pages/WhatsAppLanding';
import VerifyPage from './pages/VerifyPage';
import DeepfakePage from './pages/DeepfakePage';
import TrendingPage from './pages/TrendingPage';
import './index.css';

function App() {
  const [articles, setArticles] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [currentFilter, setCurrentFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  
  // Dark mode is default.
  const [isLightMode, setIsLightMode] = useState(false);

  useEffect(() => {
    if (isLightMode) {
      document.body.classList.add('light-mode');
    } else {
      document.body.classList.remove('light-mode');
    }
  }, [isLightMode]);

  const toggleTheme = () => setIsLightMode(!isLightMode);

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/stats`);
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error("Failed to fetch stats", err);
    }
  };

  const fetchArticles = async (targetPage = 1) => {
    setLoading(true);
    try {
      let url = new URL('/api/articles', API_BASE_URL);
      url.searchParams.append('page', targetPage.toString());
      url.searchParams.append('limit', '40');
      
      if (currentFilter === 'real') url.searchParams.append('is_fake', 'false');
      else if (currentFilter === 'fake') url.searchParams.append('is_fake', 'true');
      else if (currentFilter !== 'all') url.searchParams.append('category', currentFilter);
      
      if (search) url.searchParams.append('search', search);

      const res = await fetch(url);
      const data = await res.json();
      
      setArticles(data.items || []);
      setPage(targetPage);
      setTotalPages(data.pages || 0);
      
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      console.error("Failed to fetch articles", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  useEffect(() => {
    const delayDebounceFn = setTimeout(() => {
      fetchArticles(1);
    }, 300);

    return () => clearTimeout(delayDebounceFn);
  }, [currentFilter, search]);

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages && newPage !== page) {
      fetchArticles(newPage);
    }
  };

  const renderPagination = () => {
    if (totalPages <= 1) return null;

    const pages = [];
    let start = Math.max(1, page - 2);
    let end = Math.min(totalPages, start + 4);
    
    if (end - start < 4) {
      start = Math.max(1, end - 4);
    }

    return (
      <div className="pagination-container">
        <button 
          className="page-nav" 
          disabled={page === 1}
          onClick={() => handlePageChange(page - 1)}
        >
          &larr; Prev
        </button>
        
        {start > 1 && (
          <>
            <button className="page-number" onClick={() => handlePageChange(1)}>1</button>
            {start > 2 && <span className="page-dots" style={{ color: 'var(--text-muted)' }}>...</span>}
          </>
        )}

        {Array.from({ length: (end - start) + 1 }, (_, i) => start + i).map(p => (
          <button 
            key={p} 
            className={`page-number ${page === p ? 'active' : ''}`}
            onClick={() => handlePageChange(p)}
          >
            {p}
          </button>
        ))}

        {end < totalPages && (
          <>
            {end < totalPages - 1 && <span className="page-dots" style={{ color: 'var(--text-muted)' }}>...</span>}
            <button className="page-number" onClick={() => handlePageChange(totalPages)}>{totalPages}</button>
          </>
        )}

        <button 
          className="page-nav" 
          disabled={page === totalPages}
          onClick={() => handlePageChange(page + 1)}
        >
          Next &rarr;
        </button>
      </div>
    );
  };

  return (
    <div className="app-container">
      <Navbar isLightMode={isLightMode} toggleTheme={toggleTheme} />
      <Sidebar stats={stats} currentFilter={currentFilter} setCurrentFilter={setCurrentFilter} />
      
      <main className="main-content">
        <Routes>
          <Route path="/" element={
            <>
              <div className="stats-grid">
                <div className="stat-card">
                  <span className="stat-label">Total Analyzed</span>
                  <span className="stat-value">{stats?.total_articles || 0}</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Verified Authentic</span>
                  <span className="stat-value" style={{ color: 'var(--success)' }}>{stats?.real_articles || 0}</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Flagged Misinfo</span>
                  <span className="stat-value" style={{ color: 'var(--danger)' }}>{stats?.fake_articles || 0}</span>
                </div>
              </div>

              <div className="articles-header">
                <h2>
                  {currentFilter === 'all' && 'Global Intelligence Feed'}
                  {currentFilter === 'real' && 'Verified Authentic Intelligence'}
                  {currentFilter === 'fake' && 'Flagged Misinformation'}
                  {!['all', 'real', 'fake'].includes(currentFilter) && `Top Headlines: ${currentFilter}`}
                </h2>
                
                <div className="search-bar">
                  <input 
                    type="text" 
                    className="search-input" 
                    placeholder="Search intelligence..." 
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                  <Search className="search-icon" size={16} />
                </div>
              </div>

              {loading ? (
                <div className="articles-grid">
                  {[...Array(6)].map((_, i) => (
                    <div className="skeleton-card" key={i}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                        <div className="skeleton-shimmer" style={{ width: '40%', height: '12px' }} />
                        <div className="skeleton-shimmer" style={{ width: '25%', height: '24px', borderRadius: '6px' }} />
                      </div>
                      <div>
                        <div className="skeleton-shimmer" style={{ width: '90%', height: '18px', marginBottom: '8px' }} />
                        <div className="skeleton-shimmer" style={{ width: '75%', height: '18px', marginBottom: '16px' }} />
                        <div className="skeleton-shimmer" style={{ width: '100%', height: '14px', marginBottom: '6px' }} />
                        <div className="skeleton-shimmer" style={{ width: '85%', height: '14px', marginBottom: '6px' }} />
                        <div className="skeleton-shimmer" style={{ width: '60%', height: '14px' }} />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--border-light)' }}>
                        <div className="skeleton-shimmer" style={{ width: '30%', height: '12px' }} />
                        <div className="skeleton-shimmer" style={{ width: '20%', height: '20px', borderRadius: '4px' }} />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <>
                  <div className="articles-grid">
                    {articles.length > 0 ? (
                      articles.map(article => (
                        <ArticleCard key={article.id} article={article} />
                      ))
                    ) : (
                      <div style={{ textAlign: 'center', padding: '100px', color: 'var(--text-muted)', width: '100%', gridColumn: '1 / -1' }}>
                        <p style={{ fontSize: '1.2rem', fontWeight: '500' }}>No intelligence found for the selected criteria.</p>
                      </div>
                    )}
                  </div>
                  
                  {renderPagination()}
                </>
              )}
            </>
          } />
          
          <Route path="/discord" element={<DiscordLanding />} />
          <Route path="/whatsapp" element={<WhatsAppLanding />} />
          <Route path="/verify" element={<VerifyPage />} />
          <Route path="/deepfake" element={<DeepfakePage />} />
          <Route path="/trending" element={<TrendingPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
