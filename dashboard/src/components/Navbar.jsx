import React from 'react';
import { NavLink } from 'react-router-dom';
import { Sun, Moon, Link, Smartphone, ShieldCheck, ScanFace, TrendingUp } from 'lucide-react';

const Navbar = ({ isLightMode, toggleTheme }) => {
  return (
    <nav className="navbar">
      <div className="navbar-integrations">
        <NavLink to="/discord" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <Link size={16} /> Discord Bot
        </NavLink>
        <NavLink to="/whatsapp" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <Smartphone size={16} /> WhatsApp Bot
        </NavLink>
        <NavLink to="/verify" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <ShieldCheck size={16} /> Verify URL
        </NavLink>
        <NavLink to="/deepfake" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <ScanFace size={16} /> Deepfake Detector
        </NavLink>
        <NavLink to="/trending" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <TrendingUp size={16} /> Trending Scanner
        </NavLink>
      </div>
      
      <button 
        className="theme-toggle" 
        onClick={toggleTheme} 
        title={isLightMode ? "Switch to Dark Mode" : "Switch to Light Mode"}
      >
        {isLightMode ? <Moon size={18} /> : <Sun size={18} />}
      </button>
    </nav>
  );
};

export default Navbar;
