(() => {
  const adjust = () => {
    const root = document.getElementById('joplin-plugin-content');
    const content = document.getElementById('jarvis-model-switch');
    if (root && content) {
      const contentHeight = content.getBoundingClientRect().height;
      const padding = 32; // matches 16px top/bottom padding in CSS
      const desired = Math.ceil(contentHeight + padding);
      root.style.height = `${desired}px`;
      root.style.minHeight = '0px';
      root.style.display = 'inline-block';
    }
    for (const elem of [document.body, document.documentElement]) {
      elem.style.height = 'auto';
      elem.style.minHeight = '0px';
      elem.style.display = 'inline-block';
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', adjust);
  } else {
    adjust();
  }

  window.setTimeout(adjust, 0);
})();

