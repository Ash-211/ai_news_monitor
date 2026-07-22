import React, { useState, useEffect } from 'react';
import './WhatsAppLanding.css';

const WhatsAppLanding = () => {
  const [botNumber, setBotNumber] = useState(null);
  const [isAvailable, setIsAvailable] = useState(false);
  const [loading, setLoading] = useState(true);

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  useEffect(() => {
    const fetchWhatsAppInfo = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/whatsapp-info`);
        const data = await res.json();
        setBotNumber(data.bot_number);
        setIsAvailable(data.available);
      } catch (err) {
        console.error("Failed to fetch WhatsApp info", err);
      } finally {
        setLoading(false);
      }
    };

    fetchWhatsAppInfo();
  }, []);

  const whatsappUrl = botNumber ? `https://wa.me/${botNumber}?text=hello` : '#';
  const qrCodeUrl = botNumber ? `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=https://wa.me/${botNumber}?text=hello` : null;

  return (
    <div className="whatsapp-page">
      <div className="whatsapp-hero">
        <div className="whatsapp-logo-glow">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="currentColor" className="whatsapp-svg">
            <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z" />
          </svg>
        </div>
        <h1>Chat with DailyNewsAI on WhatsApp</h1>
        <p className="hero-subtitle">Get AI-curated news briefings, credibility checks, and smart search — all from your WhatsApp. Just send a message to get started.</p>
        
        {loading ? (
          <div className="hero-loading">Loading...</div>
        ) : !isAvailable ? (
          <div className="hero-error">WhatsApp bot is currently unavailable. Please try again later.</div>
        ) : (
          <div className="hero-actions-container">
            <div className="hero-actions">
              <a href={whatsappUrl} target="_blank" rel="noopener noreferrer" className="btn-whatsapp-invite">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z" />
                </svg>
                💬 Chat on WhatsApp
              </a>
            </div>
            {qrCodeUrl && (
              <div className="whatsapp-qr-section">
                <img src={qrCodeUrl} alt="WhatsApp QR Code" className="whatsapp-qr" />
                <p>Scan to start chatting</p>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="features-section">
        <h2>Features & Commands</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon-wrapper">
              <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9.5a2.5 2.5 0 00-2.5-2.5H15"></path>
              </svg>
            </div>
            <h3>Latest News</h3>
            <p>Get the latest AI-curated news delivered instantly. Just type 'news' and receive the top articles with credibility scores.</p>
            <div className="command-tag">news</div>
          </div>

          <div className="feature-card">
            <div className="feature-icon-wrapper">
              <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
              </svg>
            </div>
            <h3>Smart Search</h3>
            <p>Search through our intelligence database by keyword, topic, or category. Find exactly what you're looking for.</p>
            <div className="command-tag">search &lt;topic&gt;</div>
          </div>

          <div className="feature-card">
            <div className="feature-icon-wrapper">
              <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
              </svg>
            </div>
            <h3>Verify News</h3>
            <p>Paste any news article URL and get an instant AI-powered credibility assessment with detailed analysis.</p>
            <div className="command-tag">verify &lt;url&gt;</div>
          </div>
        </div>
      </div>

      <div className="setup-guide-section">
        <h2>Setup & Configuration</h2>
        <div className="timeline">
          <div className="timeline-item">
            <div className="timeline-badge">1</div>
            <div className="timeline-panel">
              <h4>Save the Number</h4>
              <p>Save our WhatsApp number to your contacts or scan the QR code above to start a chat.</p>
            </div>
          </div>

          <div className="timeline-item">
            <div className="timeline-badge">2</div>
            <div className="timeline-panel">
              <h4>Say Hello</h4>
              <p>Send 'hello' or 'hi' to the bot. You'll receive a welcome message with available commands.</p>
            </div>
          </div>

          <div className="timeline-item">
            <div className="timeline-badge">3</div>
            <div className="timeline-panel">
              <h4>Stay Informed</h4>
              <p>Use commands like 'news', 'search AI', or 'categories' to explore. Tap interactive buttons for easy navigation.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WhatsAppLanding;
