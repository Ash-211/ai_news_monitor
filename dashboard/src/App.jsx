import React, { useState, useEffect, useRef, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ArticleCard from './components/ArticleCard';
// AnalyzePanel removed as requested
import './index.css';

function App() {
  const [articles, setArticles] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [currentFilter, setCurrentFilter] = useState('all'); // all, real, fake, or category
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);

  // Cache: key = "filter|search|page" → response data
  const pageCache = useRef({});
  const abortRef = useRef(null); // cancel stale fetches

  const buildCacheKey = (filter, q, p) => `${filter}|${q}|${p}`;

  const buildUrl = (filter, q, p, lim = 20) => {
    let url = new URL('http://localhost:8000/api/articles');
    url.searchParams.append('page', p.toString());
    url.searchParams.append('limit', lim.toString());
    if (filter === 'real') url.searchParams.append('is_fake', 'false');
    else if (filter === 'fake') url.searchParams.append('is_fake', 'true');
    else if (filter !== 'all') url.searchParams.append('category', filter);
    if (q) url.searchParams.append('search', q);
    return url;
  };

  const fetchStats = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/stats');
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error("Failed to fetch stats", err);
    }
  };

  // Prefetch a page silently into cache (no loading state change)
  const prefetchPage = useCallback(async (filter, q, p) => {
    if (p < 1) return;
    const key = buildCacheKey(filter, q, p);
    if (pageCache.current[key]) return; // already cached
    try {
      const res = await fetch(buildUrl(filter, q, p));
      if (!res.ok) return;
      const data = await res.json();
      pageCache.current[key] = data;
    } catch (_) {}
  }, []);

  const fetchArticles = useCallback(async (targetPage = 1, filter = currentFilter, q = search) => {
    const key = buildCacheKey(filter, q, targetPage);

    // Serve instantly from cache if available
    if (pageCache.current[key]) {
      const cached = pageCache.current[key];
      setArticles(cached.items || []);
      setPage(targetPage);
      setTotalPages(cached.pages || 0);
      setLoading(false);
      window.scrollTo({ top: 0, behavior: 'smooth' });
      prefetchPage(filter, q, targetPage + 1);
      prefetchPage(filter, q, targetPage - 1);
      return;
    }

    // Cancel any previous in-flight request
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    try {
      const res = await fetch(buildUrl(filter, q, targetPage), { signal: controller.signal });
      const data = await res.json();
      pageCache.current[key] = data;
      setArticles(data.items || []);
      setPage(targetPage);
      setTotalPages(data.pages || 0);
      window.scrollTo({ top: 0, behavior: 'smooth' });

      // Prefetch neighbours
      prefetchPage(filter, q, targetPage + 1);
      prefetchPage(filter, q, targetPage - 1);
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error("Failed to fetch articles", err);
      }
    } finally {
      setLoading(false);
    }
  }, [currentFilter, search, prefetchPage]);

  useEffect(() => {
    fetchStats();
  }, []);

  useEffect(() => {
    // Clear cache when filter or search changes
    pageCache.current = {};
    const delayDebounceFn = setTimeout(() => {
      fetchArticles(1, currentFilter, search);
    }, 400); // Reduced debounce — API is fast enough now
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
      </main>
    </div>
  );
}

export default App;
