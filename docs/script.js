'use strict';

document.documentElement.classList.add('js');

const tablist = document.querySelector('[role="tablist"]');
const tabs = [...tablist.querySelectorAll('[role="tab"]')];
const panels = [...document.querySelectorAll('.result-panel')];
function activateTab(tab, focus = false) {
  for (const item of tabs) {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
  }
  for (const panel of panels) panel.hidden = panel.id !== tab.getAttribute('aria-controls');
  if (focus) tab.focus();
}
tablist.hidden = false;
activateTab(tabs[0]);
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      activateTab(tabs[next], true);
    }
  });
});

const dialog = document.getElementById('figure-dialog');
const dialogImage = document.getElementById('dialog-image');
const dialogTitle = document.getElementById('figure-dialog-title');
let lastFigure = null;
if (typeof dialog.showModal === 'function') {
  document.querySelectorAll('[data-enlarge]').forEach(link => {
    link.addEventListener('click', event => {
      event.preventDefault();
      lastFigure = link;
      dialogImage.src = link.href;
      dialogImage.alt = link.querySelector('img').alt;
      dialogTitle.textContent = link.dataset.title;
      dialog.showModal();
      document.body.classList.add('figure-open');
    });
  });
  document.getElementById('close-dialog').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.classList.remove('figure-open');
    lastFigure?.focus({ preventScroll: true });
  });
}

const copyStatus = document.getElementById('copy-status');
document.querySelectorAll('[data-copy]').forEach(button => {
  button.hidden = false;
  button.addEventListener('click', async () => {
    const code = document.getElementById(button.dataset.copy);
    const original = button.textContent;
    try {
      await navigator.clipboard.writeText(code.textContent.trim());
      button.textContent = 'Copied';
      copyStatus.textContent = button.dataset.copy === 'bibtex' ? 'BibTeX copied to clipboard.' : 'Training command copied to clipboard.';
      window.setTimeout(() => { button.textContent = original; }, 1800);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Text selected';
      copyStatus.textContent = 'Clipboard access is unavailable. The text is selected; press Ctrl+C or Command+C to copy.';
      window.setTimeout(() => { button.textContent = original; }, 3000);
    }
  });
});
