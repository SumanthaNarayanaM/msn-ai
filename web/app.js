/**
 * MSN AI v0.2 — frontend application.
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
  getIngestionHealth,
  listFiles,
  getFile,
  uploadFile,
  analyzeFile,
  rememberFile,
  forgetFile,
  deleteFile,
  getExtractedText,
  fileContentUrl,
  fileThumbnailUrl,
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
    desc: 'Send documents, images and video so MSN AI can reason over your own files.',
    icon: 'i-upload',
    accent: 'var(--pu)',
    glow: 'linear-gradient(135deg, rgba(168,85,247,.9), rgba(236,72,153,.9))',
    live: true,
    foot: 'Live now',
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

  /* --- v0.2 files --- */
  /** @type {'explore'|'files'} Which top-level view is on screen. */
  view: 'explore',
  /** @type {Array<object>} FileRecord objects from GET /files. */
  files: [],
  /** @type {Array<object>} Files queued for upload / being polled. */
  uploading: [],
  /** @type {Array<object>} Files attached to the next message. */
  attachments: [],
  /** Capability report from GET /ingestion/health. */
  capabilities: null,
  ingestionEnabled: false,
  useKnowledge: true,
  /** Poll timers keyed by file id. */
  pollers: new Map(),
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

  /* --- v0.2 files --- */
  filesView: $('filesView'),
  filesList: $('filesList'),
  filesEmpty: $('filesEmpty'),
  filesSummary: $('filesSummary'),
  capsRow: $('capsRow'),
  dropzone: $('dropzone'),
  dropHint: $('dropHint'),
  dropProgress: $('dropProgress'),
  dropBar: $('dropBar'),
  dropLabel: $('dropLabel'),
  uploadBtn: $('uploadBtn'),
  uploadBtnMini: $('uploadBtnMini'),
  filePicker: $('filePicker'),
  attachPicker: $('attachPicker'),
  attachBtn: $('attachBtn'),
  attachmentBar: $('attachmentBar'),
  knowledgeToggle: $('knowledgeToggle'),
  textModal: $('textModal'),
  textModalTitle: $('textModalTitle'),
  textModalBody: $('textModalBody'),
  textModalIntro: $('textModalIntro'),
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
    el.appVersion.textContent = `v${String(data.version || '0.2').replace(/^v/, '')}`;
    el.modelName.textContent = model;
    el.modelProvider.textContent = `${provider} · ${data.environment || 'local'}`;
    el.offlineDetail.innerHTML = `Reconnected — ${escapeHtml(provider)} / ${escapeHtml(model)} is serving requests.`;
    fillKv(el.settingsKv, [
      ['Status', `<span style="color:#7ff0c4">● ${escapeHtml(data.status)}</span>`],
      ['App', escapeHtml(`${data.app || 'MSN AI'} v${data.version || '0.2.0'}`)],
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
      ['Streaming', 'Not available in v0.2'],
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
  clearAttachments();
  localStorage.removeItem(STORAGE_SESSION);
  renderThread();
  renderRecent();
  showWelcomeView();
  el.composerInput.value = '';
  autoResize();
  syncSendState();
  el.composerInput.focus();
  closeSidebarOnMobile();
  // Session-scoped files disappear with the chat; remembered ones do not.
  loadFiles();
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

  // Files still being processed cannot be cited, so they are reported instead
  // of being silently sent.
  const pendingAttachments = state.attachments.filter((f) => f.status === 'processing');
  const usable = state.attachments.filter((f) => f.status === 'ready');
  if (pendingAttachments.length) {
    toast(
      'Still reading files',
      `${pendingAttachments.length} file(s) are still being processed; they are not used in this reply.`,
      'warn'
    );
  }

  state.messages.push({
    role: 'user',
    content: message,
    at: new Date(),
    files: state.attachments.map((f) => ({ display_name: f.display_name, status: f.status })),
  });
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
    const data = await sendChat({
      message,
      sessionId: state.sessionId,
      attachments: usable.map((f) => f.id),
      useKnowledge: state.useKnowledge,
      signal: controller.signal,
    });
    const elapsed = Math.round(performance.now() - started);
    pending.remove();

    // The backend is the source of truth for the session id.
    state.sessionId = data.session_id;
    localStorage.setItem(STORAGE_SESSION, state.sessionId);
    // The first message can create the session, so the file list is re-read
    // against the id the backend just assigned.
    loadFiles();

    state.messages.push({
      role: 'assistant',
      content: String(data.reply ?? ''),
      at: new Date(),
      elapsed,
      sources: Array.isArray(data.sources) ? data.sources : [],
    });
    state.lastFailed = null;
    clearAttachments();
    renderThread();
    scrollToBottom();

    const secs = (elapsed / 1000).toFixed(1);
    const cited = Array.isArray(data.sources) ? data.sources.length : 0;
    toast(
      'Reply ready',
      cited ? `${data.provider} · ${data.model} · ${secs}s · ${cited} source(s)` : `${data.provider} · ${data.model} · ${secs}s`,
      'ok',
      3200
    );
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
    const files = Array.isArray(msg.files) ? msg.files : [];
    const fileChips = files.length
      ? `<div class="attachments" style="padding:0 0 8px">${files
          .map(
            (f) =>
              `<span class="chip-file" data-status="${escapeHtml(f.status || 'ready')}">
                 <span class="chip-file__name">${escapeHtml(f.display_name || f.filename || 'file')}</span>
               </span>`
          )
          .join('')}</div>`
      : '';
    body = `<div class="msg__body">${fileChips}<div class="msg__text">${escapeHtml(
      msg.content
    )}</div></div>`;
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
    // Citations: only the excerpts the backend actually used, never invented.
    const sources = Array.isArray(msg.sources) ? msg.sources : [];
    if (sources.length) target.appendChild(sourcesNode(sources));
  }
  return node;
}

function sourcesNode(sources) {
  const wrap = document.createElement('div');
  wrap.className = 'sources';
  const list = sources
    .map(
      (s) => `
      <button type="button" class="source" data-file-id="${escapeHtml(s.file_id || '')}">
        <span class="source__name">${svgIcon('i-doc')} ${escapeHtml(s.source || s.filename || '')}</span>
        <span class="source__excerpt">${escapeHtml(String(s.text || '').slice(0, 220))}</span>
      </button>`
    )
    .join('');
  wrap.innerHTML = `<span class="sources__label">Sources used</span><div class="sources__list">${list}</div>`;
  return wrap;
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

function setView(view) {
  state.view = view;
  if (el.filesView) el.filesView.hidden = view !== 'files';
  document.querySelectorAll('.nav__item[data-nav]').forEach((b) =>
    b.classList.toggle('is-active', b.dataset.nav === view)
  );
}

function showChatView() {
  const hasMessages = state.messages.length > 0;
  setView('explore');
  el.welcomeView.hidden = hasMessages;
  el.welcomeView.classList.toggle('is-active', !hasMessages);
  el.chatView.hidden = !hasMessages;
  el.thread.classList.toggle('is-active', hasMessages);
}

function showFilesView() {
  setView('files');
  closeSidebarOnMobile();
  loadFiles();
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
      toast(`${soon.dataset.soon} — Coming soon`, 'Not available in MSN AI v0.2. No backend endpoint exists for this yet.', 'warn', 5200);
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
      toast(`${soon.dataset.soon} — Coming soon`, 'No backend endpoint supports this in v0.2.', 'warn', 4600);
    }
  });

  document.querySelectorAll('.nav__item').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (btn.dataset.nav === 'prompts') {
        el.promptsModal.showModal();
        return;
      }
      if (btn.dataset.nav === 'files') {
        showFilesView();
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

  /* --- v0.2: My Files --- */
  el.uploadBtn?.addEventListener('click', () => el.filePicker?.click());
  el.uploadBtnMini?.addEventListener('click', () => el.filePicker?.click());
  el.dropzone?.addEventListener('click', () => el.filePicker?.click());
  el.dropzone?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      el.filePicker?.click();
    }
  });
  el.filePicker?.addEventListener('change', (e) => {
    handleFiles(e.target.files);
    e.target.value = '';
  });

  el.attachBtn?.addEventListener('click', () => el.attachPicker?.click());
  el.attachPicker?.addEventListener('change', (e) => {
    handleFiles(e.target.files, { attach: true });
    e.target.value = '';
  });

  el.attachmentBar?.addEventListener('click', (e) => {
    const remove = e.target.closest('[data-remove-attachment]');
    if (remove) removeAttachment(remove.dataset.removeAttachment);
  });

  el.knowledgeToggle?.addEventListener('change', (e) => {
    state.useKnowledge = e.target.checked;
    toast(
      state.useKnowledge ? 'Files enabled' : 'Files disabled',
      state.useKnowledge
        ? 'MSN AI will consult your files and cite them.'
        : 'MSN AI will answer from the model only.',
      'info',
      2600
    );
  });

  el.filesList?.addEventListener('click', (e) => {
    const trigger = e.target.closest('[data-file-action]');
    if (trigger) onFileAction(trigger.dataset.fileAction, trigger.dataset.id);
  });

  // A cited source opens the file it came from.
  el.threadInner.addEventListener('click', (e) => {
    const source = e.target.closest('.source');
    if (!source) return;
    state.view = 'files';
    showFilesView();
    requestAnimationFrame(() => {
      const card = el.filesList?.querySelector(`[data-id="${CSS.escape(source.dataset.fileId || '')}"]`);
      card?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card?.animate(
        [{ outline: '2px solid var(--cy)' }, { outline: '2px solid transparent' }],
        { duration: 1600 }
      );
    });
  });

  // Drag and drop, with the whole window guarded so stray drops never navigate away.
  ['dragenter', 'dragover'].forEach((type) =>
    window.addEventListener(type, (e) => {
      e.preventDefault();
      if (state.ingestionEnabled && el.dropzone) el.dropzone.classList.add('is-over');
    })
  );
  ['dragleave', 'drop'].forEach((type) =>
    window.addEventListener(type, (e) => {
      e.preventDefault();
      if (e.type === 'dragleave' && e.relatedTarget) return;
      el.dropzone?.classList.remove('is-over');
    })
  );
  window.addEventListener('drop', (e) => {
    const inFilesView = state.view === 'files' || e.target.closest?.('#dropzone, #attachPicker, .composer');
    if (e.dataTransfer?.files?.length && inFilesView && !e.target.closest?.('#dropzone')) {
      handleFiles(e.dataTransfer.files, { attach: !!e.target.closest?.('.composer') });
    }
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
  renderFiles();
  renderAttachments();
  showWelcomeView();
  renderRecent();
  bindEvents();
  autoResize();
  syncSendState();
  setBusy(false);

  (async () => {
    const online = await refreshHealth();
    if (online) await restoreSession();
    // Capability discovery is independent of the session, so files can be
    // listed (and uploads accepted) before the first message is ever sent.
    loadIngestion();
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

/* ================================================================== */
/* v0.2 — My Files                                                     */
/* ================================================================== */

const CAP_LABELS = {
  ocr: 'Text in images (OCR)',
  vision: 'Image understanding',
  speech: 'Video speech-to-text',
  video_frames: 'Video frame sampling',
  embeddings: 'Semantic search',
};

const STATUS_LABELS = {
  uploaded: 'Uploaded',
  processing: 'Reading…',
  ready: 'Ready',
  failed: 'Failed',
};

/** Load capabilities and the file list; safe to call repeatedly. */
async function loadIngestion() {
  try {
    const health = await getIngestionHealth();
    state.capabilities = health.capabilities || {};
    state.ingestionEnabled = Boolean(health.enabled);
    renderCapabilities(health);
  } catch (err) {
    state.capabilities = {};
    state.ingestionEnabled = false;
    renderCapabilities(null, err);
  }
  await loadFiles();
}

async function loadFiles() {
  try {
    const data = await listFiles({ sessionId: state.sessionId });
    state.files = Array.isArray(data.files) ? data.files : [];
  } catch (err) {
    if (!(err instanceof ApiError && err.isOffline)) {
      toast('Could not load files', err.message, 'warn');
    }
    state.files = [];
  }
  renderFiles();
}

function renderCapabilities(health, error) {
  if (!el.capsRow) return;
  if (error) {
    el.capsRow.innerHTML = `<span class="cap" data-available="false">
      <span class="cap__dot"></span>File reading is unavailable — ${escapeHtml(error.message)}
    </span>`;
    return;
  }
  const caps = state.capabilities || {};
  const parts = Object.entries(CAP_LABELS).map(([key, label]) => {
    const cap = caps[key] || {};
    const available = Boolean(cap.available);
    // The backend's own reason text is shown on hover, so nothing is guessed.
    const reason = available ? 'Available' : cap.reason || 'Not installed';
    return `<span class="cap" data-available="${available}" title="${escapeHtml(reason)}">
      <span class="cap__dot"></span>${escapeHtml(label)}${available ? '' : ' <em style="opacity:.7">— not installed</em>'}
    </span>`;
  });
  const limit = health?.limits?.max_upload_mb;
  if (limit) {
    parts.push(
      `<span class="cap" data-available="true" title="Uploads larger than this are rejected">
        <span class="cap__dot"></span>Up to ${escapeHtml(String(limit))} MB per file
      </span>`
    );
  }
  el.capsRow.innerHTML = parts.join('');
  if (el.dropHint) {
    el.dropHint.textContent = state.ingestionEnabled
      ? `or click to browse. Up to ${limit ?? 50} MB per file, processed on this machine.`
      : 'File ingestion is disabled on the backend right now.';
  }
}

function fileIconFor(file) {
  if (file.has_thumbnail) return null; // an <img> is rendered instead
  const ext = (file.extension || '').toLowerCase();
  if (['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.svg'].includes(ext)) return 'i-image';
  if (['.mp4', '.mov', '.mkv', '.webm', '.avi'].includes(ext)) return 'i-video';
  if (['.xlsx', '.xlsm', '.csv', '.tsv'].includes(ext)) return 'i-tools';
  if (['.zip'].includes(ext)) return 'i-doc';
  return 'i-doc';
}

function humanSize(bytes) {
  if (!bytes && bytes !== 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = Number(bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

function renderFiles() {
  if (!el.filesList) return;
  const rows = [...state.uploading, ...state.files];
  el.filesEmpty.hidden = rows.length > 0;

  el.filesList.innerHTML = rows.map(fileCardHtml).join('');

  if (state.files.length) {
    const ready = state.files.filter((f) => f.status === 'ready').length;
    el.filesSummary.textContent = `${state.files.length} file(s) · ${ready} readable · indexed locally for cited answers.`;
  } else if (!state.uploading.length) {
    el.filesSummary.textContent =
      'Upload documents, images and video. MSN AI reads them locally and cites what it used.';
  }
}

function fileCardHtml(file) {
  const status = file.status || 'uploaded';
  const icon = fileIconFor(file);
  const thumb = file.has_thumbnail
    ? `<img src="${fileThumbnailUrl(file.id)}" alt="" loading="lazy">`
    : svgIcon(icon || 'i-doc');

  const bits = [humanSize(file.size), STATUS_LABELS[status] || status];
  if (file.chunk_count) bits.push(`${file.chunk_count} passage(s)`);
  if (file.is_video) bits.push('video');
  if (file.is_image) bits.push('image');

  const progress =
    status === 'processing' || status === 'uploaded'
      ? `<div class="bar" style="margin-top:6px"><div class="bar__fill" style="width:${Math.round(
          (file.progress || 0.05) * 100
        )}%"></div></div>${
          file.progress_stage
            ? `<span class="filecard__unavail">${escapeHtml(file.progress_stage)}</span>`
            : ''
        }`
      : '';

  // Anything the backend could not read is stated plainly, never hidden.
  const problems = [];
  if (file.error) problems.push(`<span class="filecard__error">${escapeHtml(file.error)}</span>`);
  (file.warnings || []).slice(0, 2).forEach((w) => {
    problems.push(`<span class="filecard__warn">${escapeHtml(w)}</span>`);
  });
  Object.entries(file.unavailable || {}).forEach(([k, v]) => {
    problems.push(`<span class="filecard__unavail">${escapeHtml(k)}: ${escapeHtml(String(v))}</span>`);
  });

  const scopePill =
    file.remembered && file.scope !== 'session'
      ? `<span class="pill" data-scope="${escapeHtml(file.scope)}" title="Visible in every session">${escapeHtml(
          file.scope
        )}</span>`
      : '';

  const actions = [];
  if (status === 'ready') {
    actions.push(
      `<button type="button" class="mini" data-file-action="text" data-id="${escapeHtml(
        file.id
      )}">View text</button>`
    );
  }
  if (status === 'failed' || (file.warnings || []).length) {
    actions.push(
      `<button type="button" class="mini" data-file-action="analyze" data-id="${escapeHtml(
        file.id
      )}" title="Re-run extraction, e.g. after installing a missing tool">Re-read</button>`
    );
  }
  if (status === 'ready') {
    actions.push(
      file.remembered
        ? `<button type="button" class="mini" data-file-action="forget" data-id="${escapeHtml(
            file.id
          )}" title="Remove from the knowledge base but keep the file">Forget</button>`
        : `<button type="button" class="mini" data-file-action="remember" data-id="${escapeHtml(
            file.id
          )}" title="Keep this in memory across sessions">Remember</button>`
    );
  }
  actions.push(
    `<a class="mini" href="${fileContentUrl(file.id)}" download="${escapeHtml(
      file.display_name
    )}" title="Download the original file">Open</a>`,
    `<button type="button" class="mini mini--danger" data-file-action="delete" data-id="${escapeHtml(
      file.id
    )}">Delete</button>`
  );

  return `
    <div class="filecard" data-id="${escapeHtml(file.id)}" data-status="${escapeHtml(status)}">
      <div class="filecard__thumb">${thumb}</div>
      <div class="filecard__main">
        <div class="filecard__name" title="${escapeHtml(file.display_name)}">${escapeHtml(
    file.display_name || file.filename
  )}</div>
        <div class="filecard__meta">
          <span class="pill" data-status="${escapeHtml(status)}">${escapeHtml(
    STATUS_LABELS[status] || status
  )}</span>
          ${scopePill}
          <span>${escapeHtml(bits.filter(Boolean).join(' · '))}</span>
        </div>
        ${progress}
        ${problems.join('')}
      </div>
      <div class="filecard__actions">${actions.join('')}</div>
    </div>`;
}

/* ---------- upload ---------- */

async function handleFiles(fileList, { attach = false } = {}) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  if (!state.ingestionEnabled) {
    toast('File reading is off', 'The backend reports that file ingestion is disabled.', 'warn');
    return;
  }

  for (const file of files) {
    // A local placeholder appears immediately so the UI feels responsive.
    const placeholder = {
      id: `pending-${Math.random().toString(36).slice(2, 10)}`,
      display_name: file.name,
      filename: file.name,
      size: file.size,
      status: 'processing',
      progress: 0.02,
      progress_stage: 'uploading',
      is_image: file.type.startsWith('image/'),
      is_video: file.type.startsWith('video/'),
      warnings: [],
      unavailable: {},
      _local: file,
    };
    if (attach) state.attachments.push(placeholder);
    else state.uploading.push(placeholder);
    renderFiles();
    renderAttachments();
  }

  for (const placeholder of [...state.uploading, ...state.attachments]) {
    if (placeholder._local && !placeholder._sent) {
      await uploadOne(placeholder, { attach });
    }
  }
  renderFiles();
  renderAttachments();
}

async function uploadOne(placeholder, { attach }) {
  const file = placeholder._local;
  if (!file) return;
  placeholder._sent = true;
  setDropProgress(true, 0, `Uploading ${file.name}…`);
  try {
    const record = await uploadFile({
      file,
      sessionId: state.sessionId,
      onProgress: (fraction) => setDropProgress(true, fraction, `Uploading ${file.name}…`),
    });
    mergeUploaded(placeholder, record, { attach });
    if (!attach) {
      toast('Uploaded', `${record.display_name} is being read.`, 'ok', 2600);
    }
    pollUntilTerminal(record.id, { attach });
  } catch (err) {
    placeholder.status = 'failed';
    placeholder.error = err.message;
    placeholder._sent = false;
    placeholder._local = null;
    toast('Upload failed', err.message, 'error', 6000);
  } finally {
    setDropProgress(false);
    renderFiles();
    renderAttachments();
  }
}

function mergeUploaded(placeholder, record, { attach }) {
  const list = attach ? state.attachments : state.uploading;
  const index = list.findIndex((f) => f.id === placeholder.id);
  const next = { ...placeholder, ...record, _local: null };
  if (index >= 0) list.splice(index, 1, next);
  else list.push(next);
  if (attach) renderAttachments();
  else renderFiles();
}

/** Poll GET /files/{id} until the file is ready or failed. */
function pollUntilTerminal(fileId, { attach = false } = {}) {
  if (state.pollers.has(fileId)) return;
  const started = Date.now();
  // Bounded so a wedged backend cannot poll forever.
  const maxWaitMs = 15 * 60 * 1000;
  const tick = async () => {
    if (Date.now() - started > maxWaitMs) {
      state.pollers.delete(fileId);
      return;
    }
    let record;
    try {
      record = await getFile(fileId);
    } catch {
      state.pollers.set(fileId, setTimeout(tick, 2000));
      return;
    }
    const attached = state.attachments.find((f) => f.id === fileId);
    if (attached) {
      Object.assign(attached, record);
      renderAttachments();
    }
    const uploading = state.uploading.find((f) => f.id === fileId);
    if (uploading) {
      Object.assign(uploading, record);
      renderFiles();
    }
    const known = state.files.find((f) => f.id === fileId);
    if (known) Object.assign(known, record);
    else if (record.status === 'ready' || record.status === 'failed') state.files.push(record);
    renderFiles();

    if (record.status === 'ready' || record.status === 'failed') {
      state.pollers.delete(fileId);
      if (record.status === 'ready') {
        toast('Ready to use', `${record.display_name} can now be cited in answers.`, 'ok', 3000);
      } else {
        toast('Could not read file', record.error || 'Extraction failed.', 'error', 6000);
      }
      return;
    }
    state.pollers.set(fileId, setTimeout(tick, 1200));
  };
  state.pollers.set(fileId, setTimeout(tick, 600));
}

/* ---------- attachments ---------- */

function renderAttachments() {
  if (!el.attachmentBar) return;
  el.attachmentBar.hidden = state.attachments.length === 0;
  el.attachmentBar.innerHTML = state.attachments
    .map(
      (f) => `<span class="chip-file" data-status="${escapeHtml(f.status || 'ready')}"
        title="${escapeHtml(f.status === 'ready' ? 'Will be used to answer' : 'Not usable yet')}">
        <span class="chip-file__name">${escapeHtml(f.display_name || f.filename)}</span>
        <button type="button" class="chip-file__x" data-remove-attachment="${escapeHtml(
          f.id
        )}" aria-label="Remove ${escapeHtml(f.display_name || 'file')}">×</button>
      </span>`
    )
    .join('');
}

function removeAttachment(fileId) {
  state.attachments = state.attachments.filter((f) => f.id !== fileId);
  renderAttachments();
}

function clearAttachments() {
  state.attachments = [];
  renderAttachments();
}

function setDropProgress(visible, fraction = 0, label = '') {
  if (!el.dropProgress) return;
  el.dropProgress.hidden = !visible;
  if (el.dropBar) el.dropBar.style.width = `${Math.round(fraction * 100)}%`;
  if (label && el.dropLabel) el.dropLabel.textContent = label;
}

/* ---------- file actions ---------- */

async function onFileAction(action, fileId) {
  const record = state.files.find((f) => f.id === fileId) || state.attachments.find((f) => f.id === fileId);
  try {
    if (action === 'text') {
      const data = await getExtractedText(fileId);
      el.textModalTitle.textContent = data.filename || 'Extracted text';
      el.textModalIntro.textContent = data.char_count
        ? `${data.char_count.toLocaleString()} characters were extracted and indexed from this file.`
        : 'No text was extracted from this file.';
      el.textModalBody.textContent = data.text || '(empty)';
      el.textModal.showModal();
      return;
    }
    if (action === 'analyze') {
      toast('Re-reading file', 'Re-running extraction…', 'info', 2000);
      const updated = await analyzeFile(fileId);
      replaceFile(updated);
      renderFiles();
      renderAttachments();
      toast('Re-read finished', `${updated.display_name}: ${STATUS_LABELS[updated.status]}`, 'ok', 3000);
      return;
    }
    if (action === 'remember') {
      const updated = await rememberFile(fileId, { scope: 'personal' });
      replaceFile(updated);
      renderFiles();
      toast('Remembered', `${updated.display_name} stays available in all your chats.`, 'ok', 3200);
      return;
    }
    if (action === 'forget') {
      const updated = await forgetFile(fileId);
      replaceFile(updated);
      renderFiles();
      toast('Forgotten', 'The file is kept, but MSN AI will no longer cite it.', 'ok', 3200);
      return;
    }
    if (action === 'delete') {
      if (!window.confirm(`Delete "${record?.display_name || 'this file'}" and everything indexed from it?`)) {
        return;
      }
      const button = document.querySelector(`[data-file-action="delete"][data-id="${CSS.escape(fileId)}"]`);
      if (button) {
        button.disabled = true;
        button.textContent = 'Deleting…';
      }
      await deleteFile(fileId);
      state.files = state.files.filter((f) => f.id !== fileId);
      state.attachments = state.attachments.filter((f) => f.id !== fileId);
      state.pollers.delete(fileId);
      renderFiles();
      renderAttachments();
      toast('Deleted', 'The file and its indexed passages are gone.', 'ok', 3000);
    }
  } catch (err) {
    toast('That did not work', err.message, 'error', 6000);
    renderFiles();
  }
}

function replaceFile(record) {
  const index = state.files.findIndex((f) => f.id === record.id);
  if (index >= 0) state.files.splice(index, 1, record);
  else state.files.push(record);
  const attached = state.attachments.find((f) => f.id === record.id);
  if (attached) Object.assign(attached, record);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
