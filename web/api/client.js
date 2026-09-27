/**
 * MSN AI — centralized API client.
 *
 * This module is the ONLY place in the frontend that knows about backend URLs.
 * Every request goes through `request()` which provides:
 *   - JSON encode/decode
 *   - per-request timeouts (AbortController + timer)
 *   - external AbortController cancellation
 *   - HTTP error handling with the backend's {error:{code,message,status}} shape
 *   - FastAPI {detail:[...]} validation error handling
 *   - network / offline error handling
 *
 * Endpoint contract (verified against the live FastAPI backend):
 *   GET    /health          -> {status, app, version, environment, provider, model, uptime_seconds}
 *   POST   /chat            -> {session_id, reply, model, provider, usage, created_at}
 *   GET    /sessions        -> {count, sessions:[id], items:[{id,title,created_at,updated_at,message_count}]}
 *   GET    /sessions/{id}   -> {session_id, title, created_at, updated_at, message_count, messages:[{role,content}]}
 *   DELETE /sessions/{id}   -> {deleted: id}
 *
 * NOTE: the backend has no session-create endpoint. A session is created
 * implicitly by POST /chat when `session_id` is omitted. There is also no
 * streaming and no server-side cancellation in v0.1.
 */

export const API_BASE = 'http://127.0.0.1:8000/api/v1';

/** Per-endpoint timeouts (ms). Local LLM generation needs a long budget. */
export const TIMEOUTS = {
  health: 5000,
  sessions: 8000,
  session: 8000,
  chat: 180000,
};

/** Error type that carries enough context for the UI to react correctly. */
export class ApiError extends Error {
  constructor(message, { status = 0, code = 'unknown', kind = 'network', cause = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.kind = kind; // 'timeout' | 'abort' | 'http' | 'network' | 'parse'
    this.cause = cause;
  }

  /** True when the request never reached a working backend. */
  get isOffline() {
    return this.kind === 'network' || this.kind === 'timeout';
  }

  get isCancelled() {
    return this.kind === 'abort';
  }
}

/**
 * Perform a JSON request against the MSN AI backend.
 *
 * @param {string} path        Path relative to API_BASE, e.g. '/health'
 * @param {object} [options]
 * @param {'GET'|'POST'|'DELETE'} [options.method='GET']
 * @param {object}  [options.body]        JSON request body
 * @param {number}  [options.timeoutMs]   Abort after N ms (default 15000)
 * @param {AbortSignal} [options.signal]  External cancellation signal
 * @returns {Promise<any>} Parsed JSON payload
 * @throws {ApiError}
 */
export async function request(path, options = {}) {
  const {
    method = 'GET',
    body = null,
    timeoutMs = 15000,
    signal = null,
  } = options;

  const url = `${API_BASE}${path}`;

  // Controller used for the timeout; `signal` (if any) is chained into it.
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  const forwardAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener('abort', forwardAbort, { once: true });
  }

  let response;
  try {
    response = await fetch(url, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
      cache: 'no-store',
      mode: 'cors',
    });
  } catch (err) {
    if (timedOut) {
      throw new ApiError(`Request timed out after ${Math.round(timeoutMs / 1000)}s`, {
        kind: 'timeout',
        code: 'timeout',
        cause: err,
      });
    }
    if (signal && signal.aborted) {
      throw new ApiError('Request cancelled', { kind: 'abort', code: 'aborted', cause: err });
    }
    throw new ApiError(
      'Cannot reach the MSN AI backend. Is it running on 127.0.0.1:8000?',
      { kind: 'network', code: 'network_error', cause: err }
    );
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', forwardAbort);
  }

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (err) {
      if (response.ok) {
        throw new ApiError('Backend returned a non-JSON response.', {
          status: response.status,
          kind: 'parse',
          code: 'invalid_json',
          cause: err,
        });
      }
    }
  }

  if (!response.ok) {
    throw new ApiError(describeHttpError(response.status, payload), {
      status: response.status,
      kind: 'http',
      code: extractCode(payload) ?? `http_${response.status}`,
    });
  }

  if (payload === null && text) {
    throw new ApiError('Backend returned an empty response.', {
      status: response.status,
      kind: 'parse',
      code: 'empty_response',
    });
  }

  return payload;
}

/* ------------------------------------------------------------------ */
/* Endpoint wrappers                                                   */
/* ------------------------------------------------------------------ */

/** GET /health — source of truth for the online/offline indicator. */
export function getHealth({ signal } = {}) {
  return request('/health', { method: 'GET', timeoutMs: TIMEOUTS.health, signal });
}

/**
 * POST /chat — send one message to the real local model.
 *
 * Omit `sessionId` to let the backend create a new conversation session.
 * `stream` is deliberately never sent: the backend raises NotImplementedError.
 */
export function sendChat({ message, sessionId = null, signal } = {}) {
  const body = { message, stream: false };
  if (sessionId) body.session_id = sessionId;
  return request('/chat', { method: 'POST', body, timeoutMs: TIMEOUTS.chat, signal });
}

/** GET /sessions — real conversation list from the backend. */
export function listSessions({ signal } = {}) {
  return request('/sessions', { method: 'GET', timeoutMs: TIMEOUTS.sessions, signal });
}

/** GET /sessions/{id} — real message history for a conversation. */
export function getSession(sessionId, { signal } = {}) {
  return request(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'GET',
    timeoutMs: TIMEOUTS.session,
    signal,
  });
}

/** DELETE /sessions/{id} */
export function deleteSession(sessionId, { signal } = {}) {
  return request(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
    timeoutMs: TIMEOUTS.sessions,
    signal,
  });
}

/* ------------------------------------------------------------------ */
/* Error formatting                                                    */
/* ------------------------------------------------------------------ */

function extractCode(payload) {
  if (payload && typeof payload === 'object' && payload.error && payload.error.code) {
    return payload.error.code;
  }
  return null;
}

function describeHttpError(status, payload) {
  if (payload && typeof payload === 'object') {
    // MSN AI error envelope: {error: {code, message, status}}
    if (payload.error && typeof payload.error.message === 'string') {
      return payload.error.message;
    }
    // FastAPI validation envelope: {detail: [...] | "string"}
    if (Array.isArray(payload.detail)) {
      return payload.detail
        .map((d) => `${(d.loc || []).slice(1).join('.') || 'body'}: ${d.msg}`)
        .join('; ');
    }
    if (typeof payload.detail === 'string') return payload.detail;
  }
  const map = {
    404: 'The requested MSN AI resource was not found.',
    422: 'The backend rejected the request as invalid.',
    502: 'The local model failed to generate a response.',
    503: 'The MSN AI backend is not ready yet.',
  };
  return map[status] || `Backend responded with HTTP ${status}.`;
}
