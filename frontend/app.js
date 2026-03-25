/* ============================================================
   Gaffer's Guide — frontend logic
   Calls GET /api/v1/coach/advice and renders insight cards.
   ============================================================ */

const API_BASE = 'http://localhost:8000';
const ADVICE_ENDPOINT = `${API_BASE}/api/v1/coach/advice`;

/* ── DOM refs ───────────────────────────────────────────────── */
const analyzeBtn       = document.getElementById('analyze-btn');
const btnIconPlay      = document.getElementById('btn-icon-play');
const btnIconSpin      = document.getElementById('btn-icon-spin');
const btnLabel         = document.getElementById('btn-label');
const adviceContainer  = document.getElementById('advice-container');
const emptyState       = document.getElementById('empty-state');
const skeletonState    = document.getElementById('skeleton-state');
const errorBanner      = document.getElementById('error-banner');
const errorTitle       = document.getElementById('error-title');
const errorDetail      = document.getElementById('error-detail');
const pipelineBar      = document.getElementById('pipeline-bar');
const pipeRule         = document.getElementById('pipe-rule');
const pipeRag          = document.getElementById('pipe-rag');
const pipeLlm          = document.getElementById('pipe-llm');
const pipeTs           = document.getElementById('pipe-ts');
const statusPill       = document.getElementById('status-pill');
const statusDot        = document.getElementById('status-dot');
const statusLabel      = document.getElementById('status-label');

/* ── Severity config ────────────────────────────────────────── */
const SEVERITY_META = {
  critical: { cls: 'severity-critical', icon: '🚨', label: 'Critical' },
  high:     { cls: 'severity-high',     icon: '⚠️', label: 'High'     },
  medium:   { cls: 'severity-medium',   icon: '⚡', label: 'Medium'   },
  low:      { cls: 'severity-low',      icon: 'ℹ️',  label: 'Low'      },
};

const ROLE_BADGE_COLORS = [
  'bg-sky-900/70 text-sky-300 border-sky-700',
  'bg-violet-900/70 text-violet-300 border-violet-700',
  'bg-teal-900/70 text-teal-300 border-teal-700',
  'bg-fuchsia-900/70 text-fuchsia-300 border-fuchsia-700',
  'bg-amber-900/70 text-amber-300 border-amber-700',
];

/* ── Helpers ────────────────────────────────────────────────── */
function severityMeta(severity = '') {
  return SEVERITY_META[severity.toLowerCase()] ?? SEVERITY_META.medium;
}

function roleBadgeColor(index) {
  return ROLE_BADGE_COLORS[index % ROLE_BADGE_COLORS.length];
}

/** Render pipeline status badges from the API's `pipeline` object. */
function renderPipelineBar(pipeline = {}, generatedAt = '') {
  function statusCls(val = '') {
    if (val === 'ok' || val.startsWith('ok')) return 'bg-emerald-900/70 border-emerald-700 text-emerald-300';
    if (val.startsWith('skipped'))            return 'bg-slate-800 border-slate-600 text-slate-400';
    if (val === 'pending')                    return 'bg-amber-900/70 border-amber-700 text-amber-300';
    return 'bg-red-900/70 border-red-700 text-red-300';
  }

  [
    [pipeRule, pipeline.rule_engine, 'Rule Engine'],
    [pipeRag,  pipeline.rag_synthesizer, 'RAG Synth'],
    [pipeLlm,  pipeline.llm, 'LLM'],
  ].forEach(([el, val, name]) => {
    const v = val ?? '–';
    el.className = `px-2.5 py-1 rounded-full border text-xs font-medium ${statusCls(v)}`;
    el.textContent = `${name}: ${v}`;
  });

  if (generatedAt) {
    const d = new Date(generatedAt);
    pipeTs.textContent = `Generated ${d.toLocaleTimeString()}`;
  }

  pipelineBar.classList.remove('hidden');
  pipelineBar.classList.add('flex');
}

/** Build a single insight card DOM element. */
function buildCard(item, index) {
  const meta     = severityMeta(item.severity);
  const roles    = item.fc25_player_roles ?? [];
  const hasLlm   = Boolean(item.tactical_instruction);
  const hasError = Boolean(item.llm_error);
  const teamLabel = item.team === 'team_0' ? 'Team A' : 'Team B';

  /* role badges */
  const rolesBadgeHtml = roles.length
    ? roles.map((role, i) =>
        `<span class="inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${roleBadgeColor(i)}">
          <svg viewBox="0 0 6 6" class="w-1.5 h-1.5 fill-current opacity-70"><circle cx="3" cy="3" r="3"/></svg>
          ${escHtml(role)}
        </span>`
      ).join('')
    : '<span class="text-xs text-slate-600 italic">No FC 25 roles specified</span>';

  /* main instruction / LLM text */
  let instructionHtml;
  if (hasLlm) {
    /* Preserve numbered steps with visual formatting */
    const formatted = escHtml(item.tactical_instruction)
      .replace(/(\d+\.)\s+/g, '<span class="text-emerald-400 font-bold mr-1">$1</span> ')
      .replace(/\*\*(.+?)\*\*/g, '<strong class="text-white">$1</strong>');
    instructionHtml = `<p class="text-sm leading-relaxed text-slate-300">${formatted}</p>`;
  } else if (hasError) {
    instructionHtml = `
      <div class="flex items-center gap-2 text-xs text-red-400/80">
        <svg viewBox="0 0 16 16" fill="currentColor" class="w-3.5 h-3.5 shrink-0">
          <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1Zm-.75 4.25a.75.75 0 0 1 1.5 0v3a.75.75 0 0 1-1.5 0v-3Zm.75 6.5a.875.875 0 1 1 0-1.75.875.875 0 0 1 0 1.75Z"/>
        </svg>
        <span>LLM error: ${escHtml(item.llm_error)}</span>
      </div>`;
  } else {
    instructionHtml = `
      <div class="flex items-center gap-2 text-xs text-slate-500 italic">
        <svg viewBox="0 0 20 20" fill="currentColor" class="w-3.5 h-3.5 shrink-0 text-slate-600">
          <path fill-rule="evenodd" d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-7-4a1 1 0 1 1-2 0 1 1 0 0 1 2 0ZM9 9a.75.75 0 0 0 0 1.5h.253a.25.25 0 0 1 .244.304l-.459 2.066A1.75 1.75 0 0 0 10.747 15H11a.75.75 0 0 0 0-1.5h-.253a.25.25 0 0 1-.244-.304l.459-2.066A1.75 1.75 0 0 0 9.253 9H9Z" clip-rule="evenodd"/>
        </svg>
        Set <code class="bg-pitch-700 rounded px-1 text-slate-400">LLM_API_KEY</code> to unlock coaching text.
      </div>`;
  }

  const card = document.createElement('article');
  card.style.animationDelay = `${index * 80}ms`;
  card.style.opacity = '0';
  card.className = `card-enter rounded-2xl border border-pitch-600 bg-pitch-800 p-5 flex flex-col gap-4 hover:border-pitch-500 transition-colors duration-200`;

  card.innerHTML = `
    <!-- Card header -->
    <div class="flex items-start justify-between gap-3">
      <div class="flex-1 min-w-0">
        <!-- Flaw -->
        <div class="flex items-center gap-2 mb-1">
          <span class="${meta.cls} text-xs font-bold border rounded-full px-2.5 py-0.5 tracking-wide uppercase">
            ${meta.icon} ${escHtml(item.severity)}
          </span>
          <span class="text-xs text-slate-500 font-medium">${escHtml(teamLabel)} · Frame ${item.frame_idx}</span>
        </div>
        <h3 class="text-base font-bold text-white leading-snug">${meta.icon} ${escHtml(item.flaw)}</h3>
      </div>
    </div>

    <!-- Evidence -->
    <div class="flex items-start gap-2 rounded-xl bg-pitch-700/60 border border-pitch-600/50 px-3.5 py-2.5">
      <svg viewBox="0 0 20 20" fill="currentColor" class="w-3.5 h-3.5 text-slate-500 mt-0.5 shrink-0">
        <path d="M10 12.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"/>
        <path fill-rule="evenodd" d="M.664 10.59a1.651 1.651 0 0 1 0-1.186A10.004 10.004 0 0 1 10 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0 1 10 17c-4.257 0-7.893-2.66-9.336-6.41Z" clip-rule="evenodd"/>
      </svg>
      <p class="text-xs text-slate-400 leading-relaxed">${escHtml(item.evidence)}</p>
    </div>

    <!-- Philosophy -->
    <div class="flex items-center gap-2 text-xs text-slate-500">
      <svg viewBox="0 0 20 20" fill="currentColor" class="w-3.5 h-3.5 text-slate-600 shrink-0">
        <path fill-rule="evenodd" d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm.75-13a.75.75 0 0 0-1.5 0v5c0 .414.336.75.75.75h4a.75.75 0 0 0 0-1.5h-3.25V5Z" clip-rule="evenodd"/>
      </svg>
      <span class="italic">Philosophy: <strong class="text-slate-400 not-italic">${escHtml(item.matched_philosophy_author)}</strong></span>
    </div>

    <!-- Tactical instruction -->
    <div class="rounded-xl bg-pitch-900/80 border border-pitch-600/40 px-4 py-3.5 flex-1">
      <p class="text-[10px] font-bold uppercase tracking-widest text-slate-600 mb-2">Coaching Instruction</p>
      ${instructionHtml}
    </div>

    <!-- FC 25 role badges -->
    ${roles.length || true ? `
    <div>
      <p class="text-[10px] font-bold uppercase tracking-widest text-slate-600 mb-2">FC 25 Player Roles</p>
      <div class="flex flex-wrap gap-1.5">${rolesBadgeHtml}</div>
    </div>` : ''}
  `;

  return card;
}

/* ── HTML sanitiser ─────────────────────────────────────────── */
function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ── UI state helpers ───────────────────────────────────────── */
function setLoading(loading) {
  analyzeBtn.disabled  = loading;
  btnIconPlay.classList.toggle('hidden', loading);
  btnIconSpin.classList.toggle('hidden', !loading);
  btnLabel.textContent = loading ? 'Analyzing…' : 'Analyze Match';

  emptyState.classList.add('hidden');
  skeletonState.classList.toggle('hidden', !loading);
  errorBanner.classList.add('hidden');

  if (!loading) {
    skeletonState.classList.add('hidden');
  }

  /* Status pill */
  statusPill.classList.remove('hidden');
  statusPill.classList.add('flex');
  if (loading) {
    statusDot.className = 'w-2 h-2 rounded-full bg-amber-400 animate-pulse inline-block';
    statusLabel.textContent = 'Processing…';
  } else {
    statusDot.className = 'w-2 h-2 rounded-full bg-emerald-400 inline-block';
    statusLabel.textContent = 'Up to date';
  }
}

function showError(title, detail) {
  errorTitle.textContent  = title;
  errorDetail.textContent = detail;
  errorBanner.classList.remove('hidden');
  errorBanner.classList.add('flex');
  statusDot.className   = 'w-2 h-2 rounded-full bg-red-400 inline-block';
  statusLabel.textContent = 'Error';
}

/* ── Main fetch + render ────────────────────────────────────── */
async function analyzeMatch() {
  setLoading(true);
  adviceContainer.innerHTML = '';

  try {
    const response = await fetch(ADVICE_ENDPOINT, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        detail = body?.detail?.message ?? body?.detail ?? detail;
      } catch { /* non-JSON body */ }
      throw new Error(detail);
    }

    const data = await response.json();
    const items = data.advice_items ?? [];

    renderPipelineBar(data.pipeline ?? {}, data.generated_at ?? '');

    if (items.length === 0) {
      adviceContainer.innerHTML = `
        <div class="col-span-full flex flex-col items-center justify-center py-20 text-center text-slate-500">
          <svg viewBox="0 0 48 48" class="w-12 h-12 mb-4 text-slate-700 stroke-current fill-none stroke-[1.5]">
            <circle cx="24" cy="24" r="20"/>
            <path d="M24 16v8M24 32h.01"/>
          </svg>
          <p class="text-sm font-medium text-slate-400">No tactical flaws flagged</p>
          <p class="text-xs mt-1">The team's shape looks clean — no triggers fired.</p>
        </div>`;
    } else {
      items.forEach((item, i) => {
        adviceContainer.appendChild(buildCard(item, i));
      });
    }

  } catch (err) {
    showError(
      'Could not reach the coaching engine',
      err.message + ' — make sure uvicorn is running on port 8000.',
    );
  } finally {
    setLoading(false);
  }
}

/* ── Event listeners ────────────────────────────────────────── */
analyzeBtn.addEventListener('click', analyzeMatch);
