import React from 'react';
import './DiscordLanding.css';

const DiscordLanding = () => {
  // A placeholder Discord invite URL. In a real app, client_id would be configured
  const inviteUrl = "https://discord.com/api/oauth2/authorize?client_id=1512066207690391653&permissions=2147483648&scope=bot%20applications.commands";

  return (
    <div className="discord-page">
      <div className="discord-hero">
        <div className="discord-logo-glow">
          <svg width="64" height="64" viewBox="0 0 127.14 96.36" fill="currentColor" className="discord-svg">
            <path d="M107.7,8.07A105.15,105.15,0,0,0,77.26,0a77.19,77.19,0,0,0-3.3,6.83A96.67,96.67,0,0,0,53.22,6.83,77.19,77.19,0,0,0,49.88,0,105.15,105.15,0,0,0,19.44,8.07C3.66,31.58-1.95,54.65,1,77.53A105.73,105.73,0,0,0,32,96.36a77.7,77.7,0,0,0,6.63-10.85,68.43,68.43,0,0,1-10.5-5c.89-.65,1.76-1.34,2.58-2.06a75.22,75.22,0,0,0,73.1,0c.82.72,1.69,1.41,2.58,2.06a68.43,68.43,0,0,1-10.5,5,77.7,77.7,0,0,0,6.63,10.85,105.73,105.73,0,0,0,31.06-18.83C129.89,48.37,123.6,25.43,107.7,8.07ZM42.45,65.69C36.18,65.69,31,60,31,53S36.18,40.36,42.45,40.36,53.83,46,53.83,53,48.72,65.69,42.45,65.69Zm42.24,0C78.41,65.69,73.24,60,73.24,53S78.41,40.36,84.69,40.36,96.07,46,96.07,53,91,65.69,84.69,65.69Z" />
          </svg>
        </div>
        <h1>Bring DailyNewsAI to Discord</h1>
        <p className="hero-subtitle">Get AI-curated news briefings, credibility assessments, and real-time query support directly inside your Discord server.</p>
        
        <div className="hero-actions">
          <a href={inviteUrl} target="_blank" rel="noopener noreferrer" className="btn-discord-invite">
            <svg width="20" height="20" viewBox="0 0 127.14 96.36" fill="currentColor">
              <path d="M107.7,8.07A105.15,105.15,0,0,0,77.26,0a77.19,77.19,0,0,0-3.3,6.83A96.67,96.67,0,0,0,53.22,6.83,77.19,77.19,0,0,0,49.88,0,105.15,105.15,0,0,0,19.44,8.07C3.66,31.58-1.95,54.65,1,77.53A105.73,105.73,0,0,0,32,96.36a77.7,77.7,0,0,0,6.63-10.85,68.43,68.43,0,0,1-10.5-5c.89-.65,1.76-1.34,2.58-2.06a75.22,75.22,0,0,0,73.1,0c.82.72,1.69,1.41,2.58,2.06a68.43,68.43,0,0,1-10.5,5,77.7,77.7,0,0,0,6.63,10.85,105.73,105.73,0,0,0,31.06-18.83C129.89,48.37,123.6,25.43,107.7,8.07ZM42.45,65.69C36.18,65.69,31,60,31,53S36.18,40.36,42.45,40.36,53.83,46,53.83,53,48.72,65.69,42.45,65.69Zm42.24,0C78.41,65.69,73.24,60,73.24,53S78.41,40.36,84.69,40.36,96.07,46,96.07,53,91,65.69,84.69,65.69Z" />
            </svg>
            Add to Discord
          </a>
        </div>
      </div>

      <div className="features-section">
        <h2>Features & Commands</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon-wrapper">
              <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
            </div>
            <h3>Daily Briefings</h3>
            <p>Automated news updates delivered to your chosen channel every day at 3:00 PM UTC. Configurable by admins.</p>
            <div className="command-tag">/setup_daily</div>
          </div>

          <div className="feature-card">
            <div className="feature-icon-wrapper">
              <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
              </svg>
            </div>
            <h3>Smart Search</h3>
            <p>Query the entire intelligence pipeline by date or sector tag. Instant evaluation in chat.</p>
            <div className="command-tag">/news [date] [tag]</div>
          </div>

          <div className="feature-card">
            <div className="feature-icon-wrapper">
              <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16"></path>
              </svg>
            </div>
            <h3>Interactive Menus</h3>
            <p>Filter today's news feeds interactively using native Discord dropdown menus and components.</p>
            <div className="command-tag">/newscategories</div>
          </div>
        </div>
      </div>

      <div className="setup-guide-section">
        <h2>Setup & Configuration</h2>
        <div className="timeline">
          <div className="timeline-item">
            <div className="timeline-badge">1</div>
            <div className="timeline-panel">
              <h4>Invite the Bot</h4>
              <p>Click the "Add to Discord" button above. Authenticate and select your target Discord Server.</p>
            </div>
          </div>

          <div className="timeline-item">
            <div className="timeline-badge">2</div>
            <div className="timeline-panel">
              <h4>Register Daily News Channel</h4>
              <p>Navigate to the text channel where you want daily drops to appear, and run the <code>/setup_daily</code> command. (Requires Administrator permission).</p>
            </div>
          </div>

          <div className="timeline-item">
            <div className="timeline-badge">3</div>
            <div className="timeline-panel">
              <h4>Enjoy AI News Intel</h4>
              <p>The bot will automatically publish a summary of evaluated real and fake news every day at 3:00 PM. Server members can search news anytime with <code>/news</code> or trigger active drops via <code>/newsdaily</code>.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiscordLanding;
