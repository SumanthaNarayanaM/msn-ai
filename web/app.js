/**
 * MSN AI v0.1 — frontend application.
 *
 * Every backend call goes through ./api/client.js. This module owns UI state
 * and rendering only. There are no hardcoded/fake AI replies: the assistant
 * text shown in the thread always comes from POST /api/v1/chat, and the
 * online/offline indicator always comes from GET /api/v1/health.
 */

import {
  API_BASE,
  getHealth,
  listSessions,
  getSession,
  deleteSession,
  sendChat,
  ApiError,
} from './api/client.js';

/* ================================================================== */
/* Constants                                                          */
/* ================================================================== */

const STORAGE_THEME = 'msnai.theme';
const STORAGE_SESSION = 'msnai.activeSession';

const HEALTH_POLL_MS = 20000;
const MAX_SESSIONS_SHOWN = 40;
const SCROLL_BOTTOM_GAP = 120;

const CATEGORIES = ['All', 'Coding', 'Content', 'Analysis', 'Creative', 'Research', 'General'];

const QUICK_PROMPTS = [
  { text: 'Write a Java Hello World program and explain how it works.', cat: 'Coding', icon: 'i-code', accent: 'var(--cy)' },
  { text: 'Write a Python function that reverses a string, with type hints and comments.', cat: 'Coding', icon: 'i-code', accent: 'var(--cy)' },
  { text: 'Explain the difference between a Python list and a tuple.', cat: 'Coding', icon: 'i-code', accent: 'var(--cy)' },
  { text: 'Write a short product description for a specialty coffee brand.', cat: 'Content', icon: 'i-doc', accent: 'var(--pu)' },
  { text: 'Rewrite the following text so it is clearer and more concise:', cat: 'Content', icon: 'i-doc', accent: 'var(--pu)' },
  { text: 'Compare REST and GraphQL for a small team project.', cat: 'Analysis', icon: 'i-tools', accent: 'var(--bl)' },
  { text: 'Break down the pros and cons of learning Python before JavaScript.', cat: 'Analysis', icon: 'i-tools', accent: 'var(--bl)' },
  { text: 'Write a short sci-fi story opening set on a Mars colony.', cat: 'Creative', icon: 'i-sparkle', accent: 'var(--pk)' },
  { text: 'Name a sci-fi film with a similar mood to Blade Runner and explain why.', cat: 'Creative', icon: 'i-sparkle', accent: 'var(--pk)' },
  { text: 'Explain how HTTPS certificate validation works, step by step.', cat: 'Research', icon: 'i-search', accent: 'var(--gr)' },
  { text: 'What are the trade-offs of SQLite versus Postgres for a desktop app?', cat: 'Research', icon: 'i-search', accent: 'var(--gr)' },
  { text: 'Give me a realistic 4-week study plan for learning Python.', cat: 'General', icon: 'i-clock', accent: 'var(--am)' },
  { text: 'How can I stay focused while working on a long coding task?', cat: 'General', icon: 'i-clock', accent: 'var(--am)' },
];

const FEATURE_CARDS = [
  {
    title: 'Chat & Code',
    desc: 'Talk to your local model and get real answers with syntax-highlighted code.',
    icon: 'i-chat',
    accent: 'var(--cy)',
    glow: 'linear-gradient(135deg, rgba(34,211,238,.9), rgba(59,130,246,.9))',
    live: true,
    foot: 'Live now',
  },
  {
    title: 'Upload Files',
    desc: 'Send documents and images so the model can reason over your own files.',
    icon: 'i-upload',
    accent: 'var(--pu)',
    glow: 'linear-gradient(135deg, rgba(168,85,247,.9), rgba(236,72,153,.9))',
    live: false,
    foot: 'Coming soon',
  },
  {
    title: 'Voice Mode',
    desc: 'Talk to MSN AI hands-free and hear spoken replies.',
    icon: 'i-mic',
    accent: 'var(--pk)',
    glow: 'linear-gradient(135deg, rgba(236,72,153,.9), rgba(251,191,36,.9))',
    live: false,
    foot: 'Coming soon',
  },
  {
    title: 'Video Creation',
    desc: 'Turn an idea or script into a short generated clip on your machine.',
    icon: 'i-video',
    accent: 'var(--gr)',
    glow: 'linear-gradient(135deg, rgba(52,211,153,.9), rgba(34,211,238,.9))',
    live: false,
    foot: 'Coming soon',
  },
];

/* ================================================================== */
/* State                                                              */
/* ================================================================== */

const state = {
  /** 'connecting' | 'online' | 'offline' */
  status: 'connecting',
  health: null,
  healthError: null,
  healthCheckedAt: null,
  sessions: [],
  sessionId: null,
  messages: [],
  generating: false,
  abort: null,
  lastFailed: null,
  category: 'All',
  search: '',
  theme: 'dark',
};

const $ = (id) => document.getElementById(id);

const el = {
  sidebar: $('sidebar'),
  scrim: $('scrim'),
  sidebarClose: $('sidebarClose'),
  menuBtn: $('menuBtn'),
  newChatBtn: $('newChatBtn'),
  recentList: $('recentList'),
  recentEmpty: $('recentEmpty'),
  refreshSessions: $('refreshSessions'),

  statusPill: $('statusPill'),
  statusText: $('statusText'),
  sideDot: $('sideDot'),
  sideModel: $('sideModel'),
  sideProvider: $('sideProvider'),
  appVersion: $('appVersion'),

  search: $('sessionSearch'),
  themeBtn: $('themeBtn'),
  notifyBtn: $('notifyBtn'),
  avatarBtn: $('avatarBtn'),

  offlineBanner: $('offlineBanner'),
  offlineDetail: $('offlineDetail'),
  apiBaseLabel: $('apiBaseLabel'),
  retryHealth: $('retryHealth'),

  main: $('main'),
  welcomeView: $('welcomeView'),
  chatView: $('chatView'),
  thread: $('thread'),
  threadInner: $('threadInner'),
  categoryChips: $('categoryChips'),
  featureCards: $('featureCards'),
  quickPrompts: $('quickPrompts'),
  composerModel: $('composerModel'),

  composerForm: $('composerForm'),
  composer: document.querySelector('.composer'),
  composerInput: $('composerInput'),
  sendBtn: $('sendBtn'),
  stopBtn: $('stopBtn'),

  settingsBtn: $('settingsBtn'),
  settingsModal: $('settingsModal'),
  settingsKv: $('settingsKv'),
  settingsRefresh: $('settingsRefresh'),
  modelSelect: $('modelSelect'),
  modelModal: $('modelModal'),
  modelName: $('modelName'),
  modelProvider: $('modelProvider'),
  modelKv: $('modelKv'),
  modelDot: $('modelDot'),
  promptsModal: $('promptsModal'),
  promptLibrary: $('promptLibrary'),
  toasts: $('toasts'),
};

/* ================================================================== */
/* Utilities                                                          */
/* ================================================================== */

const svgIcon = (id, cls = '') =>
  `<svg class="${cls}" aria-hidden="true"><use href="#${id}"/></svg>`;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function toast(title, msg = '', type = 'info', ttl = 4600) {
  const icon = { info: 'i-sparkle', ok: 'i-check', warn: 'i-alert', error: 'i-alert' }[type] || 'i-sparkle';
  const node = document.createElement('div');
  node.className = `toast toast--${type}`;
  node.setAttribute('role', type === 'error' ? 'alert' : 'status');
  node.innerHTML = `${svgIcon(icon, 'toast__icon')}<div class="toast__body"><div class="toast__title">${escapeHtml(
    title
  )}</div>${msg ? `<div class="toast__msg">${escapeHtml(msg)}</div>` : ''}</div>`;
  el.toasts.appendChild(node);
  setTimeout(() => {
    node.classList.add('is-out');
    setTimeout(() => node.remove(), 300);
  }, ttl);
}

function formatRelative(ts) {
  if (!ts) return '';
  const seconds = Math.max(0, (Date.now() / 1000) - Number(ts));
  if (seconds < 60) return 'just now';
  const units = [
    ['m', 60, 60],
    ['h', 3600, 24],
    ['d', 86400, 7],
  ];
  for (const [label, div, max] of units) {
    if (seconds < div * max) {
      const value = Math.floor(seconds / div);
      return `${value}${label} ago`;
    }
  }
  return new Date(Number(ts) * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function formatDuration(seconds) {
  if (seconds == null) return '—';
  const s = Math.max(0, Math.round(Number(seconds)));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

const clockTime = (date) =>
  date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

/* ================================================================== */
/* Health                                                             */
/* ================================================================== */

/**
 * Ask GET /api/v1/health. Always settles into 'online' or 'offline' — the
 * client enforces a 5s timeout, so the UI can never stay on "Connecting…".
 */
async function refreshHealth({ silent = false } = {}) {
  if (!silent) setStatus('connecting');
  try {
    const data = await getHealth();
    if (!data || typeof data !== 'object' || typeof data.status !== 'string') {
      throw new ApiError('Unexpected health payload from backend.', { kind: 'parse', code: 'bad_payload' });
    }
    state.health = data;
    state.healthError = null;
    state.healthCheckedAt = Date.now();
    setStatus('online');
    applyHealth(data);
    if (state.sessionId === null) loadSessions();
    return true;
  } catch (err) {
    state.health = null;
    state.healthError = err;
    setStatus('offline');
    applyHealth(null);
    return false;
  }
}

function setStatus(status) {
  state.status = status;
  const text = { connecting: 'Connecting…', online: 'MSN AI Online', offline: 'Backend Offline' }[status];
  el.statusPill.dataset.state = status;
  el.statusText.textContent = text;
  el.sideDot.dataset.state = status;
  el.modelDot.dataset.state = status;
  el.offlineBanner.hidden = status !== 'offline';
  syncSendState();
}

/** Render everything that must reflect the *real* backend health payload. */
function applyHealth(data) {
  if (data) {
    const model = data.model || 'local model';
    const provider = data.provider || 'local';
    el.sideModel.textContent = `AI Model (${provider})`;
    el.sideProvider.textContent = model;
    el.composerModel.textContent = provider;
    el.appVersion.textContent = `v${String(data.version || '0.1').replace(/^v/, '')}`;
    el.modelName.textContent = model;
    el.modelProvider.textContent = `${provider} · ${data.environment || 'local'}`;
    el.offlineDetail.innerHTML = `Reconnected — ${escapeHtml(provider)} / ${escapeHtml(model)} is serving requests.`;
    fillKv(el.settingsKv, [
      ['Status', `<span style="color:#7ff0c4">● ${escapeHtml(data.status)}</span>`],
      ['App', escapeHtml(`${data.app || 'MSN AI'} v${data.version || '0.1.0'}`)],
      ['Provider', `<code>${escapeHtml(data.provider || '—')}</code>`],
      ['Model', `<code>${escapeHtml(data.model || '—')}</code>`],
      ['Environment', escapeHtml(data.environment || '—')],
      ['Uptime', escapeHtml(formatDuration(data.uptime_seconds))],
      ['API base', `<code>${escapeHtml(API_BASE)}</code>`],
    ]);
    fillKv(el.modelKv, [
      ['Model', `<code>${escapeHtml(data.model || '—')}</code>`],
      ['Provider', `<code>${escapeHtml(data.provider || '—')}</code>`],
      ['Runtime', 'llama.cpp (CPU)'],
      ['Context', '2048 tokens'],
      ['Streaming', 'Not available in v0.1'],
    ]);
  } else {
    const reason = state.healthError ? state.healthError.message : 'Backend unreachable.';
    el.sideModel.textContent = 'AI Model (Local)';
    el.sideProvider.textContent = 'backend offline';
    el.composerModel.textContent = 'offline';
    el.offlineDetail.innerHTML = `MSN AI cannot reach <code>${escapeHtml(
      API_BASE.replace(/^https?:\/\//, '')
    )}</code>. Start it with <code>python -m app.main</code> — ${escapeHtml(reason)}`;
    fillKv(el.settingsKv, [
      ['Status', '<span style="color:#ffb0bd">○ Backend Offline</span>'],
      ['API base', `<code>${escapeHtml(API_BASE)}</code>`],
      ['Reason', escapeHtml(reason)],
      ['Hint', 'Run <code>python -m app.main</code> in D:\\MSN AI, then press Retry.'],
    ]);
    fillKv(el.modelKv, [['Status', 'Backend offline — no model loaded']]);
  }
}

function fillKv(dl, rows) {
  dl.innerHTML = rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${v}</dd>`).join('');
}

/* ================================================================== */
/* Sessions                                                           */
/* ================================================================== */

async function loadSessions() {
  try {
    const data = await listSessions();
    const items = Array.isArray(data?.items) ? data.items : [];
    state.sessions = items
      .map((s) => ({
        id: String(s.id),
        title: s.title || 'New conversation',
        updated_at: s.updated_at,
        message_count: Number(s.message_count) || 0,
      }))
      .sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
    renderRecent();
    return true;
  } catch (err) {
    renderRecent();
    if (state.status === 'online') {
      toast('Could not load chats', err.message, 'warn');
    }
    return false;
  }
}

function renderRecent() {
  const q = state.search.trim().toLowerCase();
  const list = q ? state.sessions.filter((s) => s.title.toLowerCase().includes(q)) : state.sessions;
  const shown = list.slice(0, MAX_SESSIONS_SHOWN);

  el.recentEmpty.textContent = state.sessions.length
    ? 'No chats match your search.'
    : 'No conversations yet. Start chatting!';
  el.recentEmpty.hidden = shown.length > 0;

  el.recentList.innerHTML = shown
    .map(
      (s) => `
      <li class="recent__item">
        <button type="button" class="recent__btn${s.id === state.sessionId ? ' is-active' : ''}"
                data-session="${escapeHtml(s.id)}"
                aria-current="${s.id === state.sessionId ? 'true' : 'false'}">
          ${svgIcon('i-chat', 'recent__icon')}
          <span class="recent__body">
            <span class="recent__title-text">${escapeHtml(s.title)}</span>
            <span class="recent__meta">${escapeHtml(formatRelative(s.updated_at))} · ${
        s.message_count
      } msg</span>
          </span>
        </button>
        <button type="button" class="recent__del" data-delete="${escapeHtml(s.id)}"
                aria-label="Delete conversation: ${escapeHtml(s.title)}" title="Delete conversation">
          ${svgIcon('i-trash')}
        </button>
      </li>`
    )
    .join('');
}

async function openSession(sessionId) {
  if (state.generating) {
    toast('Busy', 'Wait for the current reply to finish, or press Stop.', 'warn');
    return;
  }
  if (sessionId === state.sessionId) {
    closeSidebarOnMobile();
    return;
  }
  try {
    const data = await getSession(sessionId);
    state.sessionId = data.session_id;
    state.messages = (Array.isArray(data.messages) ? data.messages : [])
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ role: m.role, content: String(m.content ?? ''), at: new Date() }));
    localStorage.setItem(STORAGE_SESSION, state.sessionId);
    renderThread();
    renderRecent();
    showChatView();
    closeSidebarOnMobile();
    scrollToBottom(true);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      localStorage.removeItem(STORAGE_SESSION);
      state.sessionId = null;
      await loadSessions();
      toast('Conversation missing', 'It was removed on the backend.', 'warn');
      return;
    }
    toast('Could not open chat', err.message, 'error');
  }
}

async function removeSession(sessionId) {
  const target = state.sessions.find((s) => s.id === sessionId);
  if (!window.confirm(`Delete "${target?.title || 'this conversation'}" on the backend? This cannot be undone.`)) return;
  try {
    await deleteSession(sessionId);
    if (state.sessionId === sessionId) {
      state.sessionId = null;
      state.messages = [];
      localStorage.removeItem(STORAGE_SESSION);
      renderThread();
      showWelcomeView();
    }
    await loadSessions();
    toast('Conversation deleted', 'Removed from the backend.', 'ok');
  } catch (err) {
    toast('Delete failed', err.message, 'error');
  }
}

function newChat() {
  if (state.generating) {
    toast('Busy', 'Press Stop before starting a new chat.', 'warn');
    return;
  }
  state.sessionId = null;
  state.messages = [];
  state.lastFailed = null;
  localStorage.removeItem(STORAGE_SESSION);
  renderThread();
  renderRecent();
  showWelcomeView();
  el.composerInput.value = '';
  autoResize();
  syncSendState();
  el.composerInput.focus();
  closeSidebarOnMobile();
}

/* ================================================================== */
/* Chat                                                               */
/* ================================================================== */

async function submitMessage(text) {
  const message = String(text ?? '').trim();
  if (!message || state.generating) return;

  if (state.status === 'offline') {
    toast('Backend Offline', `Start the backend, then retry. (${API_BASE})`, 'error');
  }

  state.messages.push({ role: 'user', content: message, at: new Date() });
  state.generating = true;
  state.lastFailed = { message, sessionId: state.sessionId };
  el.composerInput.value = '';
  autoResize();
  syncSendState();
  setBusy(true);
  showChatView();
  renderThread();
  scrollToBottom();

  const controller = new AbortController();
  state.abort = controller;
  const pending = appendThinking();
  const started = performance.now();

  try {
    const data = await sendChat({ message, sessionId: state.sessionId, signal: controller.signal });
    const elapsed = Math.round(performance.now() - started);
    pending.remove();

    // The backend is the source of truth for the session id.
    state.sessionId = data.session_id;
    localStorage.setItem(STORAGE_SESSION, state.sessionId);

    state.messages.push({ role: 'assistant', content: String(data.reply ?? ''), at: new Date(), elapsed });
    state.lastFailed = null;
    renderThread();
    scrollToBottom();

    const secs = (elapsed / 1000).toFixed(1);
    toast('Reply ready', `${data.provider} · ${data.model} · ${secs}s`, 'ok', 3200);
  } catch (err) {
    pending.remove();

    if (err instanceof ApiError && err.isCancelled) {
      state.messages.push({
        role: 'assistant',
        content: '_Generation stopped. The local model may still finish on the backend._',
        at: new Date(),
        muted: true,
      });
      renderThread();
      scrollToBottom();
      toast('Stopped', 'The request was cancelled in the browser.', 'warn');
    } else {
      state.messages.push({
        role: 'error',
        content: err.message,
        title: err instanceof ApiError && err.isOffline ? 'Backend Offline' : 'Could not get a reply',
        at: new Date(),
        retry: { message, sessionId: state.sessionId },
      });
      if (err instanceof ApiError && err.isOffline) setStatus('offline');
      renderThread();
      scrollToBottom();
    }
  } finally {
    state.generating = false;
    state.abort = null;
    setBusy(false);
    renderRecent();
    loadSessions();
  }
}

function retryLast() {
  const failed = state.lastFailed;
  if (!failed || state.generating) return;
  const last = state.messages[state.messages.length - 1];
  if (last && last.role === 'error') {
    state.messages = state.messages.slice(0, -1);
  }
  state.sessionId = failed.sessionId;
  submitMessage(failed.message);
}

function stopGeneration() {
  if (!state.abort) return;
  state.abort.abort();
}

function setBusy(busy) {
  el.composer.classList.toggle('is-busy', busy);
  el.stopBtn.hidden = !busy;
  el.sendBtn.disabled = busy || !el.composerInput.value.trim();
  el.composerInput.setAttribute('aria-busy', String(busy));
  if (busy) {
    el.composerInput.placeholder = 'MSN AI is thinking...';
  } else {
    el.composerInput.placeholder = 'Ask MSN AI anything...';
  }
}

function syncSendState() {
  el.sendBtn.disabled = state.generating || !el.composerInput.value.trim();
}

function autoResize() {
  const ta = el.composerInput;
  ta.style.height = 'auto';
  ta.style.height = `${Math.min(ta.scrollHeight, 208)}px`;
}

/* ================================================================== */
/* Markdown + code rendering                                          */
/* ================================================================== */

const ALLOWED_TAGS = new Set([
  'P', 'BR', 'STRONG', 'B', 'EM', 'I', 'DEL', 'S', 'CODE', 'PRE', 'UL', 'OL', 'LI',
  'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'BLOCKQUOTE', 'HR', 'A', 'TABLE', 'THEAD',
  'TBODY', 'TFOOT', 'TR', 'TH', 'TD', 'SPAN', 'DIV', 'INPUT',
]);

const ALLOWED_ATTRS = {
  A: ['href', 'title'],
  INPUT: ['type', 'checked', 'disabled'],
  TH: ['align'],
  TD: ['align'],
  CODE: ['class'],
  PRE: ['class'],
  OL: ['start'],
};

const SAFE_SCHEME = /^(https?:|mailto:|#|\/|\.\/|\.\.\/)/i;

/** Strip every tag/attribute outside a small allowlist. */
function sanitize(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  const doomed = [];

  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (!ALLOWED_TAGS.has(node.tagName)) {
      doomed.push(node);
      continue;
    }
    const allowed = ALLOWED_ATTRS[node.tagName] || [];
    for (const attr of Array.from(node.attributes)) {
      const name = attr.name.toLowerCase();
      if (name.startsWith('on') || !allowed.includes(name)) {
        node.removeAttribute(attr.name);
      }
    }
    if (node.tagName === 'A') {
      const href = node.getAttribute('href') || '';
      if (!SAFE_SCHEME.test(href.trim())) {
        node.removeAttribute('href');
      } else {
        node.setAttribute('target', '_blank');
        node.setAttribute('rel', 'noopener noreferrer nofollow');
      }
    }
    if (node.tagName === 'INPUT' && node.getAttribute('type') !== 'checkbox') {
      doomed.push(node);
    }
  }

  doomed.forEach((node) => {
    // IMG / IFRAME / SCRIPT etc. — drop the node but keep readable text.
    const text = node.tagName === 'IMG' ? (node.getAttribute('alt') || '') : node.textContent || '';
    const span = document.createElement('span');
    span.textContent = text;
    node.replaceWith(span);
  });
}

/** Wrap each `pre > code` in a labelled, copyable, highlighted block. */
function decorateCodeBlocks(root) {
  const hljs = window.hljs;

  root.querySelectorAll('pre > code').forEach((codeEl) => {
    const pre = codeEl.parentElement;
    const match = /language-([\w+#.-]+)/.exec(codeEl.className || '');
    const lang = match ? match[1].toLowerCase() : '';
    const label = lang ? (lang === 'js' ? 'javascript' : lang) : 'text';

    if (hljs && typeof hljs.highlightElement === 'function') {
      try {
        if (lang && hljs.getLanguage && hljs.getLanguage(lang)) {
          hljs.highlightElement(codeEl);
        } else {
          codeEl.removeAttribute('data-highlighted');
          hljs.highlightElement(codeEl);
        }
      } catch {
        /* highlighting is best-effort */
      }
    }

    const wrap = document.createElement('div');
    wrap.className = 'code';
    wrap.innerHTML = `
      <div class="code__bar">
        <span class="code__dots" aria-hidden="true"><i></i><i></i><i></i></span>
        <span class="code__lang">${escapeHtml(label)}</span>
        <button type="button" class="code__copy" aria-label="Copy code to clipboard">
          ${svgIcon('i-copy')}<span>Copy</span>
        </button>
      </div>`;
    pre.replaceWith(wrap);
    wrap.appendChild(pre);
  });
}

/** Build a sanitized, highlighted markdown container. */
function renderMarkdown(text) {
  const container = document.createElement('div');
  container.className = 'md';
  const source = String(text ?? '');

  if (!window.marked || typeof window.marked.parse !== 'function') {
    container.textContent = source;
    return container;
  }

  let html;
  try {
    html = window.marked.parse(source, { gfm: true, breaks: true });
  } catch {
    container.textContent = source;
    return container;
  }

  container.innerHTML = html;
  sanitize(container);
  decorateCodeBlocks(container);
  return container;
}

/* ================================================================== */
/* Thread rendering                                                   */
/* ================================================================== */

function messageNode(msg, index) {
  const isUser = msg.role === 'user';
  const isError = msg.role === 'error';
  const kind = isError ? 'error' : isUser ? 'user' : 'ai';
  const who = isError ? 'MSN AI' : isUser ? 'You' : 'MSN AI';
  const at = msg.at instanceof Date ? msg.at : new Date();

  const node = document.createElement('article');
  node.className = `msg msg--${kind}`;
  node.dataset.index = String(index);

  const avatar = isUser
    ? '<span class="msg__avatar">YOU</span>'
    : `<span class="msg__avatar">${svgIcon('i-sparkle')}</span>`;

  const head = `<div class="msg__head">
      <span class="msg__who">${escapeHtml(who)}</span>
      <span class="msg__time">${escapeHtml(clockTime(at))}${
    msg.elapsed ? ` · ${(msg.elapsed / 1000).toFixed(1)}s` : ''
  }</span>
    </div>`;

  let body;
  if (isError) {
    body = `<div class="err">
        ${svgIcon('i-alert', 'err__icon')}
        <div class="err__text">
          <div class="err__title">${escapeHtml(msg.title || 'Something went wrong')}</div>
          <div class="err__msg">${escapeHtml(msg.content)}</div>
          <div class="err__actions">
            <button type="button" class="btn btn--sm" data-action="retry">${svgIcon(
              'i-refresh'
            )}<span>Retry</span></button>
          </div>
        </div>
      </div>`;
  } else if (isUser) {
    body = `<div class="msg__body"><div class="msg__text">${escapeHtml(msg.content)}</div></div>`;
  } else {
    body = `<div class="msg__body"></div>`;
  }

  node.innerHTML = `${avatar}<div class="msg__main">${head}${body}</div>`;

  if (!isError && !isUser) {
    const target = node.querySelector('.msg__body');
    if (msg.muted) {
      target.innerHTML = `<div class="msg__text" style="color:var(--tx-3);font-style:italic">${escapeHtml(
        msg.content.replace(/^_|_$/g, '')
      )}</div>`;
    } else {
      target.appendChild(renderMarkdown(msg.content));
    }
  }
  return node;
}

function renderThread() {
  const scroll = el.main?.scrollTop ?? 0;
  const frag = document.createDocumentFragment();

  if (!state.messages.length) {
    frag.innerHTML = `<p class="recent__empty" style="text-align:center;padding:30px 0">
      No messages in this conversation yet. Ask MSN AI something below.
    </p>`;
  } else {
    state.messages.forEach((msg, i) => frag.appendChild(messageNode(msg, i)));
  }

  el.threadInner.replaceChildren(frag);
  if (scroll > 0) el.main.scrollTop = scroll;
}

function appendThinking() {
  const node = document.createElement('article');
  node.className = 'msg msg--ai';
  node.innerHTML = `
    <span class="msg__avatar">${svgIcon('i-sparkle')}</span>
    <div class="msg__main">
      <div class="msg__head"><span class="msg__who">MSN AI</span></div>
      <div class="msg__body">
        <div class="thinking" role="status" aria-live="polite">
          <span class="thinking__dots" aria-hidden="true"><i></i><i></i><i></i></span>
          <span>MSN AI is thinking...</span>
        </div>
      </div>
    </div>`;
  el.threadInner.appendChild(node);
  scrollToBottom();
  return node;
}

function scrollToBottom(instant = false) {
  requestAnimationFrame(() => {
    const main = $('main');
    if (!main) return;
    main.scrollTo({ top: main.scrollHeight, behavior: instant ? 'auto' : 'smooth' });
  });
}

function showChatView() {
  const hasMessages = state.messages.length > 0;
  el.welcomeView.hidden = hasMessages;
  el.welcomeView.classList.toggle('is-active', !hasMessages);
  el.chatView.hidden = !hasMessages;
  el.thread.classList.toggle('is-active', hasMessages);
  document.querySelectorAll('.nav__item[data-nav]').forEach((b) =>
    b.classList.toggle('is-active', b.dataset.nav === 'explore' && !hasMessages)
  );
}

function showWelcomeView() {
  showChatView();
}

/* ================================================================== */
/* Welcome content                                                    */
/* ================================================================== */

function renderWelcome() {
  el.categoryChips.innerHTML = CATEGORIES.map(
    (cat) =>
      `<button type="button" class="chip${state.category === cat ? ' is-active' : ''}" data-cat="${escapeHtml(
        cat
      )}" aria-pressed="${state.category === cat}">${escapeHtml(cat)}</button>`
  ).join('');

  el.featureCards.innerHTML = FEATURE_CARDS.map(
    (card) => `
      <${card.live ? 'button type="button"' : 'div'} class="card ${card.live ? 'card--live' : 'card--soon'}"
        ${card.live ? 'data-action="focus-composer"' : `data-soon="${escapeHtml(card.title)}"`}
        style="--card-accent:${card.accent};--card-glow:${card.glow}"
        ${card.live ? '' : 'aria-disabled="true"'}>
        <span class="card__icon">${svgIcon(card.icon)}</span>
        <span class="card__title">${escapeHtml(card.title)}</span>
        <span class="card__desc">${escapeHtml(card.desc)}</span>
        <span class="card__foot">${card.live ? svgIcon('i-check') : svgIcon('i-clock')}<span>${escapeHtml(
      card.foot
    )}</span></span>
      </${card.live ? 'button' : 'div'}>`
  ).join('');

  renderQuickPrompts();
}

function renderQuickPrompts() {
  const list =
    state.category === 'All' ? QUICK_PROMPTS : QUICK_PROMPTS.filter((p) => p.cat === state.category);
  const shown = (list.length ? list : QUICK_PROMPTS).slice(0, 6);

  el.quickPrompts.innerHTML = shown
    .map(
      (p) => `<button type="button" class="prompt" data-prompt="${escapeHtml(p.text)}" style="--prompt-accent:${p.accent}">
        ${svgIcon(p.icon, 'prompt__icon')}
        <span class="prompt__text">${escapeHtml(p.text)}</span>
      </button>`
    )
    .join('');
}

function renderPromptLibrary() {
  const groups = CATEGORIES.filter((c) => c !== 'All')
    .map((cat) => {
      const items = QUICK_PROMPTS.filter((p) => p.cat === cat);
      if (!items.length) return '';
      return `<div class="lib__group">
          <div class="lib__label">${escapeHtml(cat)}</div>
          ${items
            .map(
              (p) =>
                `<button type="button" class="prompt" data-prompt="${escapeHtml(
                  p.text
                )}" style="--prompt-accent:${p.accent}">${svgIcon(p.icon, 'prompt__icon')}<span class="prompt__text">${escapeHtml(
                  p.text
                )}</span></button>`
            )
            .join('')}
        </div>`;
    })
    .join('');
  el.promptLibrary.innerHTML = groups;
}

/** Load a prompt into the composer (never auto-sends a paid/long local generation). */
function loadPrompt(text) {
  el.composerInput.value = text;
  autoResize();
  syncSendState();
  el.composerInput.focus();
  el.composer.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  el.promptsModal.open && el.promptsModal.close();
  document.querySelectorAll('.modal[open]').forEach((m) => m.close());
  toast('Prompt loaded', 'Edit it if you like, then press Send.', 'info', 3200);
}

/* ================================================================== */
/* Sidebar (responsive)                                               */
/* ================================================================== */

function openSidebar() {
  el.sidebar.classList.add('is-open');
  el.scrim.hidden = false;
  el.menuBtn.setAttribute('aria-expanded', 'true');
  el.sidebarClose.focus();
}

function closeSidebarOnMobile() {
  if (!el.sidebar.classList.contains('is-open')) return;
  el.sidebar.classList.remove('is-open');
  el.scrim.hidden = true;
  el.menuBtn.setAttribute('aria-expanded', 'false');
}

/* ================================================================== */
/* Events                                                             */
/* ================================================================== */

function bindEvents() {
  /* --- composer --- */
  el.composerForm.addEventListener('submit', (e) => {
    e.preventDefault();
    submitMessage(el.composerInput.value);
  });

  el.composerInput.addEventListener('input', () => {
    autoResize();
    syncSendState();
  });

  el.composerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!el.sendBtn.disabled) submitMessage(el.composerInput.value);
    }
  });

  el.composerInput.addEventListener('focus', () => el.composer.classList.add('is-focus'));
  el.composerInput.addEventListener('blur', () => el.composer.classList.remove('is-focus'));
  el.stopBtn.addEventListener('click', stopGeneration);

  /* --- sidebar --- */
  el.newChatBtn.addEventListener('click', newChat);
  el.menuBtn.addEventListener('click', openSidebar);
  el.sidebarClose.addEventListener('click', closeSidebarOnMobile);
  el.scrim.addEventListener('click', closeSidebarOnMobile);
  el.refreshSessions.addEventListener('click', async () => {
    el.refreshSessions.disabled = true;
    const ok = await loadSessions();
    el.refreshSessions.disabled = false;
    toast(ok ? 'Chats refreshed' : 'Could not refresh', ok ? `${state.sessions.length} conversation(s) on the backend.` : 'Backend unreachable.', ok ? 'ok' : 'warn');
  });

  el.recentList.addEventListener('click', (e) => {
    const del = e.target.closest('[data-delete]');
    if (del) {
      e.stopPropagation();
      removeSession(del.dataset.delete);
      return;
    }
    const btn = e.target.closest('[data-session]');
    if (btn) openSession(btn.dataset.session);
  });

  /* --- topbar --- */
  el.search.addEventListener('input', () => {
    state.search = el.search.value;
    renderRecent();
  });

  el.themeBtn.addEventListener('click', toggleTheme);

  el.notifyBtn.addEventListener('click', () => {
    const h = state.health;
    toast(
      h ? 'MSN AI Online' : 'Backend Offline',
      h
        ? `${h.provider} · ${h.model} · up ${formatDuration(h.uptime_seconds)}`
        : `No connection to ${API_BASE}.`,
      h ? 'ok' : 'error'
    );
  });

  el.avatarBtn.addEventListener('click', () => {
    el.settingsModal.showModal();
  });

  el.retryHealth.addEventListener('click', async () => {
    setStatus('connecting');
    const ok = await refreshHealth();
    toast(ok ? 'MSN AI Online' : 'Backend Offline', ok ? 'Connected to the local model.' : `Could not reach ${API_BASE}.`, ok ? 'ok' : 'error');
  });

  /* --- dialogs --- */
  el.settingsBtn.addEventListener('click', () => el.settingsModal.showModal());
  el.settingsRefresh.addEventListener('click', async () => {
    el.settingsRefresh.disabled = true;
    const ok = await refreshHealth();
    el.settingsRefresh.disabled = false;
    if (ok) loadSessions();
  });
  el.modelSelect.addEventListener('click', () => el.modelModal.showModal());

  /* --- welcome interactions --- */
  el.categoryChips.addEventListener('click', (e) => {
    const chip = e.target.closest('[data-cat]');
    if (!chip) return;
    state.category = chip.dataset.cat;
    renderWelcome();
  });

  el.featureCards.addEventListener('click', (e) => {
    const soon = e.target.closest('[data-soon]');
    if (soon) {
      toast(`${soon.dataset.soon} — Coming soon`, 'Not available in MSN AI v0.1. No backend endpoint exists for this yet.', 'warn', 5200);
      return;
    }
    el.composerInput.focus();
    el.composer.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  });

  el.quickPrompts.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-prompt]');
    if (btn) loadPrompt(btn.dataset.prompt);
  });

  el.promptLibrary.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-prompt]');
    if (btn) loadPrompt(btn.dataset.prompt);
  });

  /* --- thread: copy + retry --- */
  el.threadInner.addEventListener('click', async (e) => {
    const retry = e.target.closest('[data-action="retry"]');
    if (retry) {
      retry.disabled = true;
      retryLast();
      return;
    }
    const copyBtn = e.target.closest('.code__copy');
    if (copyBtn) {
      const code = copyBtn.closest('.code')?.querySelector('pre code');
      if (!code) return;
      const text = code.innerText;
      const ok = await copyText(text);
      copyBtn.classList.toggle('is-done', ok);
      copyBtn.querySelector('span').textContent = ok ? 'Copied' : 'Failed';
      copyBtn.setAttribute('aria-label', ok ? 'Code copied' : 'Copy failed');
      setTimeout(() => {
        copyBtn.classList.remove('is-done');
        const span = copyBtn.querySelector('span');
        if (span) span.textContent = 'Copy';
      }, 2000);
    }
  });

  /* --- global "coming soon" --- */
  document.addEventListener('click', (e) => {
    const soon = e.target.closest('[data-soon]');
    if (soon && !soon.closest('.card') && !soon.classList.contains('card')) {
      toast(`${soon.dataset.soon} — Coming soon`, 'No backend endpoint supports this in v0.1.', 'warn', 4600);
    }
  });

  document.querySelectorAll('.nav__item').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (btn.dataset.nav === 'prompts') {
        el.promptsModal.showModal();
        return;
      }
      if (btn.dataset.nav === 'explore') {
        if (state.messages.length) newChat();
        else {
          showWelcomeView();
          closeSidebarOnMobile();
        }
      }
    });
  });

  /* --- keyboard --- */
  document.addEventListener('keydown', (e) => {
    const typing = ['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName);
    if (e.key === '/' && !typing) {
      e.preventDefault();
      el.search.focus();
    }
    if (e.key === 'Escape') {
      if (state.generating) stopGeneration();
      if (document.activeElement === el.search) {
        el.search.value = '';
        state.search = '';
        renderRecent();
        el.search.blur();
      }
      closeSidebarOnMobile();
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      newChat();
    }
  });

  window.addEventListener('online', () => refreshHealth());
  window.addEventListener('offline', () => setStatus('offline'));
  window.addEventListener('resize', () => {
    if (window.innerWidth > 900) closeSidebarOnMobile();
  });
}

async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext !== false) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the legacy path */
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.cssText = 'position:fixed;top:-1000px;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

/* ================================================================== */
/* Theme                                                              */
/* ================================================================== */

function applyTheme(theme) {
  state.theme = theme === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = state.theme;
  try {
    localStorage.setItem(STORAGE_THEME, state.theme);
  } catch {
    /* storage may be unavailable */
  }
}

function toggleTheme() {
  applyTheme(state.theme === 'dark' ? 'light' : 'dark');
  toast(`Theme: ${state.theme === 'dark' ? 'Dark' : 'Light'}`, 'Your choice is saved locally.', 'info', 2200);
}

/* ================================================================== */
/* Boot                                                               */
/* ================================================================== */

function boot() {
  el.apiBaseLabel.textContent = API_BASE.replace(/^https?:\/\//, '');

  try {
    const saved = localStorage.getItem(STORAGE_THEME);
    if (saved) applyTheme(saved);
  } catch {
    /* ignore */
  }

  renderWelcome();
  renderPromptLibrary();
  renderThread();
  showWelcomeView();
  renderRecent();
  bindEvents();
  autoResize();
  syncSendState();
  setBusy(false);

  (async () => {
    const online = await refreshHealth();
    if (online) await restoreSession();
    setInterval(() => {
      if (!document.hidden) refreshHealth({ silent: true });
    }, HEALTH_POLL_MS);
  })();
}

/** Restore the conversation that was active before a page refresh. */
async function restoreSession() {
  let saved = null;
  try {
    saved = localStorage.getItem(STORAGE_SESSION);
  } catch {
    return;
  }
  if (!saved) return;
  try {
    const data = await getSession(saved);
    state.sessionId = data.session_id;
    state.messages = (Array.isArray(data.messages) ? data.messages : [])
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ role: m.role, content: String(m.content ?? ''), at: new Date() }));
    renderThread();
    showChatView();
    scrollToBottom(true);
  } catch (err) {
    localStorage.removeItem(STORAGE_SESSION);
    state.sessionId = null;
    if (!(err instanceof ApiError && err.status === 404)) {
      toast('Could not restore chat', err.message, 'warn');
    }
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
