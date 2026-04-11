import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ArticleCard from './components/ArticleCard';
// AnalyzePanel removed as requested
import './index.css';

function App() {
  const [articles, setArticles] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [currentFilter, setCurrentFilter] = useState('all'); // all, real, fake, analyze, or category
  const [search, setSearch] = useState('');

  const fetchStats = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/stats');
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error("Failed to fetch stats", err);
    }
  };

  const fetchArticles = async () => {
    setLoading(true);
    try {
      let url = new URL('http://localhost:8000/api/articles');
      url.searchParams.append('limit', '50');
      
      if (currentFilter === 'real') url.searchParams.append('is_fake', 'false');
      else if (currentFilter === 'fake') url.searchParams.append('is_fake', 'true');
      else if (currentFilter !== 'all') url.searchParams.append('category', currentFilter);
      
      if (search) url.searchParams.append('search', search);

      const res = await fetch(url);
      const data = await res.json();
      setArticles(data.items || []);
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
      fetchArticles();
    }, 300); // 300ms delay for search debounce

    return () => clearTimeout(delayDebounceFn);
  }, [currentFilter, search]);


  return (
    <div className="app-container">
      <Sidebar stats={stats} currentFilter={currentFilter} setCurrentFilter={setCurrentFilter} />
      
      <main className="main-content">
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

        {loading ? (
          <div className="loader">
            <div className="spinner"></div>
          </div>
        ) : (
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
        )}
      </main>
    </div>
  );
}

export default App;
