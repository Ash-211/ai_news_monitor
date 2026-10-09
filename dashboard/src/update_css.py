import os
import re

directory = './pages'

replacements = {
    # Backgrounds and borders
    r'#ffffff': 'var(--bg-card)',
    r'white(?!-space)': 'var(--bg-card)',
    r'#f1f5f9': 'var(--bg-surface)',
    r'#f8fafc': 'var(--bg-surface)',
    r'#eff6ff': 'var(--bg-surface)',
    r'#fafafa': 'var(--bg-surface)',
    r'#e2e8f0': 'var(--border-light)',
    r'#cbd5e1': 'var(--border-strong)',
    
    # Texts
    r'#0f172a': 'var(--text-main)',
    r'#1e293b': 'var(--text-main)',
    r'#334155': 'var(--text-main)',
    r'#64748b': 'var(--text-muted)',
    r'#94a3b8': 'var(--text-muted)',
    r'#475569': 'var(--text-muted)',
    
    # Status colors
    r'#fef2f2': 'rgba(255, 69, 58, 0.05)',
    r'#fee2e2': 'rgba(255, 69, 58, 0.2)',
    r'#fca5a5': 'rgba(255, 69, 58, 0.5)',
    r'#991b1b': 'var(--danger)',
    r'#7f1d1d': '#fff',
    r'#ef4444': 'var(--danger)',
    
    r'#ecfdf5': 'rgba(50, 215, 75, 0.05)',
    r'#22c55e': 'var(--success)',
    r'#15803d': 'var(--success)',
    
    # Shadows
    r'box-shadow:\s*0\s+[^;]+;': 'box-shadow: var(--shadow-sm);',
}

for filename in os.listdir(directory):
    if filename.endswith(".css") or filename.endswith(".jsx"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        new_content = content
        for pattern, replacement in replacements.items():
            # For exact color matches we use a bit of regex boundaries
            # like r'(?i)' for case insensitivity
            if not filename.endswith('.jsx') or ('css' not in filename):
                # Apply regex for colors ignoring case
                new_content = re.sub(f'(?i){pattern}', replacement, new_content)
        
        # specific hardcoded css variables in those pages
        if filename.endswith(".css"):
            new_content = re.sub(r'var\(--text-primary[^)]*\)', 'var(--text-main)', new_content)
            new_content = re.sub(r'var\(--surface-card[^)]*\)', 'var(--bg-card)', new_content)
            new_content = re.sub(r'var\(--border-subtle[^)]*\)', 'var(--border-light)', new_content)
            new_content = re.sub(r'var\(--surface-hover[^)]*\)', 'var(--bg-card-hover)', new_content)
            new_content = re.sub(r'var\(--surface-bg[^)]*\)', 'var(--bg-body)', new_content)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f'Updated {filename}')
