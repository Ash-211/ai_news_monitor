import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ArticleCard from './components/ArticleCard';

import './index.css';

function App() {
  const [articles, setArticles] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [currentFilter, setCurrentFilter] = useState('all'); // all, real, fake, or category
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

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
      
      // Auto-scroll to top when page changes
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

  // Helper to render page buttons
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
          &larr; Previous
        </button>
        
        {start > 1 && (
          <>
            <button className="page-number" onClick={() => handlePageChange(1)}>1</button>
            {start > 2 && <span className="page-dots">...</span>}
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
            {end < totalPages - 1 && <span className="page-dots">...</span>}
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
      <Sidebar stats={stats} currentFilter={currentFilter} setCurrentFilter={setCurrentFilter} />
      
      <main className="main-content">
        <Routes>
          <Route path="/" element={
            <>
              <div className="stats-grid">
                <div className="stat-card">
                  <span className="stat-label">Total Articles Analyzed</span>
                  <span className="stat-value">{stats?.total_articles || 0}</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Verified Real News</span>
                  <span className="stat-value" style={{ color: 'var(--success)' }}>{stats?.real_articles || 0}</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Detected Fake News</span>
                  <span className="stat-value" style={{ color: 'var(--danger)' }}>{stats?.fake_articles || 0}</span>
                </div>
              </div>

              <div className="articles-header">
                <h2>
                  {currentFilter === 'all' && 'Intelligence Feed'}
                  {currentFilter === 'real' && 'Verified Authentic News'}
                  {currentFilter === 'fake' && 'Flagged Misinformation'}
                  {!['all', 'real', 'fake'].includes(currentFilter) && `Top Headlines: ${currentFilter}`}
                </h2>
                
                <div className="header-actions" style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                  <a 
                    href="https://discord.com/api/oauth2/authorize?client_id=1512066207690391653&permissions=2147483648&scope=bot%20applications.commands" 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      padding: '0.75rem 1.25rem',
                      backgroundColor: '#5865F2',
                      color: 'white',
                      textDecoration: 'none',
                      borderRadius: '12px',
                      fontWeight: '600',
                      fontSize: '0.93rem',
                      boxShadow: '0 4px 14px rgba(88, 101, 242, 0.3)',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <svg width="20" height="20" viewBox="0 0 127.14 96.36" fill="currentColor">
                      <path d="M107.7,8.07A105.15,105.15,0,0,0,77.26,0a77.19,77.19,0,0,0-3.3,6.83A96.67,96.67,0,0,0,53.22,6.83,77.19,77.19,0,0,0,49.88,0,105.15,105.15,0,0,0,19.44,8.07C3.66,31.58-1.95,54.65,1,77.53A105.73,105.73,0,0,0,32,96.36a77.7,77.7,0,0,0,6.63-10.85,68.43,68.43,0,0,1-10.5-5c.89-.65,1.76-1.34,2.58-2.06a75.22,75.22,0,0,0,73.1,0c.82.72,1.69,1.41,2.58,2.06a68.43,68.43,0,0,1-10.5,5,77.7,77.7,0,0,0,6.63,10.85,105.73,105.73,0,0,0,31.06-18.83C129.89,48.37,123.6,25.43,107.7,8.07ZM42.45,65.69C36.18,65.69,31,60,31,53S36.18,40.36,42.45,40.36,53.83,46,53.83,53,48.72,65.69,42.45,65.69Zm42.24,0C78.41,65.69,73.24,60,73.24,53S78.41,40.36,84.69,40.36,96.07,46,96.07,53,91,65.69,84.69,65.69Z" />
                    </svg>
                    Add to Discord
                  </a>
                  
                  <div className="search-bar">
                    <input 
                      type="text" 
                      className="search-input" 
                      placeholder="Search intelligence..." 
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                    />
                    <svg className="search-icon" width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                    </svg>
                  </div>
                </div>
              </div>

              {loading ? (
                <div className="loader">
                  <div className="spinner"></div>
                </div>
              ) : (
                <>
                  <div className="articles-grid">
                    {articles.length > 0 ? (
                      articles.map(article => (
                        <ArticleCard key={article.id} article={article} />
                      ))
                    ) : (
                      <div style={{ textAlign: 'center', padding: '100px', color: 'var(--text-muted)', width: '100%' }}>
                        <p style={{ fontSize: '1.2rem', fontWeight: '500' }}>No articles found for the selected criteria.</p>
                      </div>
                    )}
                  </div>
                  
                  {renderPagination()}
                </>
              )}
            </>
          } />

        </Routes>
      </main>
    </div>
  );
}

export default App;
