/* EduLens — student learning dashboard.
 * Vanilla JS single-page app. All numbers shown come from the REST API; a missing
 * value is always displayed as "not available", never replaced with a fallback score. */
(() => {
  "use strict";

  /* ================= Utilities ================= */
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const icon = (n) => `<svg class="icon" aria-hidden="true"><use href="#i-${n}"/></svg>`;
  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const pct = (v) => (isNum(v) ? `${Math.round(v)}%` : null);
  const avg = (arr) => { const a = arr.filter(isNum); return a.length ? a.reduce((x, y) => x + y, 0) / a.length : null; };
  const enc = encodeURIComponent;
  const norm = (s) => String(s || "").trim().toLowerCase();
  const fmtDate = (iso) => { const d = new Date(iso); return isNaN(d) ? "Date unknown" : d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }); };
  const fmtShort = (iso) => { const d = new Date(iso); return isNaN(d) ? "" : d.toLocaleDateString(undefined, { day: "numeric", month: "short" }); };
  const AGENTS = { chatgpt: "ChatGPT", claude: "Claude", gemini: "Gemini", cursor: "Cursor", generic: "Other AI tool" };
  const agentLabel = (a) => AGENTS[norm(a)] || (a ? String(a).charAt(0).toUpperCase() + String(a).slice(1) : "Unknown tool");
  const UNDERSTANDING = {
    understood: { label: "Understood", tone: "green", progress: "On track" },
    partial: { label: "Partly understood", tone: "amber", progress: "Making progress" },
    confused: { label: "Needs review", tone: "red", progress: "Needs review" },
  };
  const pill = (text, tone = "") => `<span class="pill ${tone}">${esc(text)}</span>`;
  const understandingPill = (cls) => (UNDERSTANDING[cls] ? pill(UNDERSTANDING[cls].label, UNDERSTANDING[cls].tone) : pill("Not analysed yet"));
  const progressPill = (cls) => (UNDERSTANDING[cls] ? pill(UNDERSTANDING[cls].progress, UNDERSTANDING[cls].tone) : pill("Not enough data yet"));

  /* ================= State ================= */
  const state = {
    studentId: localStorage.getItem("edulens_student_id") || "student_001",
    students: [],
    token: 0,
    sessionFilters: { page: 0, limit: 10, search: "", subject: "", topic: "", cls: "" },
    progressTab: "overall",
  };

  /* ================= API ================= */
  async function api(path, { params, method = "GET", body } = {}) {
    const url = new URL("/api" + path, window.location.origin);
    Object.entries(params || {}).forEach(([k, v]) => { if (v !== null && v !== undefined && v !== "") url.searchParams.set(k, v); });
    let res;
    try {
      res = await fetch(url, { method, headers: body ? { "Content-Type": "application/json" } : undefined, body: body ? JSON.stringify(body) : undefined });
    } catch {
      throw new Error("Can't reach the EduLens server. Check that it is running, then try again.");
    }
    if (!res.ok) {
      let detail = "";
      try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail || j); } catch { /* ignore */ }
      if (res.status === 401) detail = "This server requires sign-in, which this page doesn't support yet. " + detail;
      throw new Error(detail || `Something went wrong (error ${res.status}).`);
    }
    return res.json();
  }
  const withStudent = (extra = {}) => ({ student_id: state.studentId, ...extra });
  const getS = (path, extra) => api(path, { params: withStudent(extra) });

  /* ================= Shell: toasts, theme, nav ================= */
  function toast(msg, kind = "") {
    const el = document.createElement("div");
    el.className = "toast " + kind; el.textContent = msg;
    $("#toasts").appendChild(el);
    setTimeout(() => el.remove(), kind === "error" ? 7000 : 4000);
  }
  function applyTheme(pref) {
    localStorage.setItem("edulens_theme", pref);
    const dark = pref === "dark" || (pref === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => { if ((localStorage.getItem("edulens_theme") || "system") === "system") applyTheme("system"); });

  function renderIdentity() {
    $("#who-name").textContent = state.studentId;
    $("#who-avatar").textContent = state.studentId.replace(/[^a-z0-9]/gi, "").slice(0, 2).toUpperCase() || "ST";
    const sel = $("#student-select");
    const ids = Array.from(new Set([...state.students, state.studentId]));
    sel.innerHTML = ids.map((id) => `<option value="${esc(id)}" ${id === state.studentId ? "selected" : ""}>${esc(id)}</option>`).join("");
  }
  function switchStudent(id) {
    id = (id || "").trim();
    if (!id || id === state.studentId) return;
    state.studentId = id;
    localStorage.setItem("edulens_student_id", id);
    state.sessionFilters = { page: 0, limit: 10, search: "", subject: "", topic: "", cls: "" };
    if (!state.students.includes(id)) state.students.push(id);
    renderIdentity(); router();
    toast(`Now showing ${id}`);
  }
  const setDrawer = (open) => $("#app").classList.toggle("drawer-open", open);

  function setNav(route) {
    const key = route === "" ? "overview" : route;
    $$("[data-nav]").forEach((a) => (a.dataset.nav === key ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
    $("#more-btn")?.toggleAttribute("aria-current", key === "connect" || key === "settings");
    if ($("#more-btn") && (key === "connect" || key === "settings")) $("#more-btn").setAttribute("aria-current", "page");
  }

  /* ================= Rendering helpers ================= */
  const main = () => $("#main");
  function paint(token, html, { focus = true, title } = {}) {
    if (token !== state.token) return false;
    main().innerHTML = html;
    main().setAttribute("aria-busy", "false");
    if (title) document.title = `${title} — EduLens`;
    if (focus) { const h = $("h1", main()); if (h) { h.tabIndex = -1; h.focus({ preventScroll: true }); } window.scrollTo(0, 0); }
    return true;
  }
  const skeleton = (rows = 3) => `<div class="stack" aria-label="Loading"><div class="skel" style="height:34px;width:40%"></div>${Array.from({ length: rows }, () => `<div class="skel" style="height:120px"></div>`).join("")}</div>`;
  const errorBox = (e) => `<div class="error-box" role="alert"><div><strong>We couldn't load this page.</strong><div class="small">${esc(e.message || e)}</div></div><button class="btn" data-action="retry">Try again</button></div>`;
  const empty = (title, text, actions = "") => `<div class="empty"><h3>${esc(title)}</h3><p>${esc(text)}</p><div>${actions}</div></div>`;
  const ingestBtn = (cls = "btn btn-primary") => `<button class="${cls}" data-action="open-ingest">${icon("plus")} Ingest session</button>`;
  const metric = (label, value, note, emptyText) => `<div class="card metric"><span class="label">${esc(label)}</span>${value !== null && value !== undefined ? `<span class="value">${esc(value)}</span>` : `<span class="value is-empty">${esc(emptyText || "Not available")}</span>`}${note ? `<span class="note">${esc(note)}</span>` : ""}</div>`;
  const pageHead = (title, desc, actions = "") => `<div class="page-head"><div><h1>${esc(title)}</h1>${desc ? `<p>${esc(desc)}</p>` : ""}</div>${actions}</div>`;

  /* ---------- SVG line chart: gaps where data is missing ---------- */
  function lineChart(points, series, label) {
    const n = points.length;
    if (!n) return "";
    const W = 640, H = 220, pl = 46, pr = 14, pt = 12, pb = 30;
    const x = (i) => (n === 1 ? pl + (W - pl - pr) / 2 : pl + (i * (W - pl - pr)) / (n - 1));
    const y = (v) => pt + ((100 - v) * (H - pt - pb)) / 100;
    let g = [0, 25, 50, 75, 100].map((v) => `<line class="grid-line" x1="${pl}" x2="${W - pr}" y1="${y(v)}" y2="${y(v)}"/><text x="${pl - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join("");
    const step = Math.max(1, Math.ceil(n / 5));
    const anchor = (i) => (i === 0 ? "start" : i === n - 1 ? "end" : "middle");
    const shown = new Set([0, n - 1]);
    for (let i = step; i < n - 1; i += step) if (n - 1 - i >= step * 0.6) shown.add(i);
    shown.forEach((i) => { g += `<text x="${x(i)}" y="${H - 8}" text-anchor="${anchor(i)}">${esc(fmtShort(points[i].date))}</text>`; });
    const colors = { understanding_score: "var(--primary)", quiz_score: "var(--green)", overall_score: "var(--amber)" };
    let body = "";
    series.forEach((s) => {
      let d = "", dots = "";
      points.forEach((p, i) => {
        const v = p[s.key];
        if (!isNum(v)) return;                       // missing → nothing drawn, line is broken
        d += isNum(points[i - 1]?.[s.key]) ? ` L${x(i)} ${y(v)}` : ` M${x(i)} ${y(v)}`;
        dots += `<circle cx="${x(i)}" cy="${y(v)}" r="4.5" fill="${colors[s.key]}" stroke="var(--surface)" stroke-width="2"><title>${esc(p.topic || "Session")}, ${esc(fmtShort(p.date))}: ${s.name} ${Math.round(v)}%</title></circle>`;
      });
      body += `<path d="${d.trim()}" fill="none" stroke="${colors[s.key]}" stroke-width="2.5" ${s.dashed ? 'stroke-dasharray="6 5"' : ""} stroke-linejoin="round"/>${dots}`;
    });
    const legend = series.map((s) => `<span><i style="background:${colors[s.key]}"></i>${esc(s.name)}</span>`).join("");
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}" style="width:100%;height:auto">${g}${body}</svg><div class="legend">${legend}</div>`;
  }

  /* ---------- Topic aggregation from real session rows ---------- */
  function topicRows(progress, breakdown) {
    const map = new Map();
    (breakdown || []).forEach((b) => map.set(norm(b.topic), { topic: b.topic, subject: b.subject, sessions: b.sessions, latest: b.latest_understanding, u: [], q: [] }));
    (progress || []).forEach((p) => {
      const r = map.get(norm(p.topic)); if (!r) return;
      if (isNum(p.understanding_score)) r.u.push(p.understanding_score);
      if (isNum(p.quiz_score)) r.q.push(p.quiz_score);
    });
    return Array.from(map.values()).map((r) => ({ ...r, avgU: avg(r.u), avgQ: avg(r.q), quizCount: r.q.length }));
  }
  const topicHref = (t) => `#/progress/topic/${enc(t)}`;

  /* ================= Router ================= */
  const TITLES = { overview: "Overview", sessions: "My Sessions", progress: "My Progress", quizzes: "Quizzes", connect: "Connect & Ingest", settings: "Settings" };
  function router() {
    const parts = (location.hash.replace(/^#\/?/, "") || "overview").split("/").map((p) => { try { return decodeURIComponent(p); } catch { return p; } });
    const route = TITLES[parts[0]] ? parts[0] : "overview";
    const token = ++state.token;
    document.body.dataset.route = route;
    setNav(route); setDrawer(false);
    main().setAttribute("aria-busy", "true");
    main().innerHTML = skeleton();
    document.title = `${TITLES[route]} — EduLens`;
    const run = {
      overview: () => viewOverview(token),
      sessions: () => (parts[1] ? viewSessionDetail(token, parts[1]) : viewSessions(token)),
      progress: () => (parts[1] === "topic" && parts[2] ? viewTopicDetail(token, parts.slice(2).join("/")) : viewProgress(token)),
      quizzes: () => viewQuizzes(token),
      connect: () => viewConnect(token),
      settings: () => viewSettings(token),
    }[route];
    Promise.resolve(run()).catch((e) => paint(token, pageHead(TITLES[route]) + errorBox(e)));
  }

  /* ================= Overview ================= */
  async function viewOverview(token) {
    const d = await getS("/dashboard");
    const s = d.summary, n = s.total_sessions, trend = d.trend || [];
    const hour = new Date().getHours();
    const greet = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
    const head = pageHead(`${greet}, ${state.studentId}`, n ? "Here's how your studying is going and what to look at next." : "Send a study conversation to EduLens and your progress will show up here.", ingestBtn());
    if (!n) {
      return paint(token, head + `<div class="card">${empty("No study sessions yet", "Learn a topic with ChatGPT, Claude or another AI tool, then send the conversation to EduLens. It takes about a minute.", ingestBtn() + ` <a class="btn" href="#/connect">How to connect</a>`)}</div>`, { title: "Overview" });
    }
    const scoredU = trend.filter((p) => isNum(p.understanding_score));
    const quizCount = trend.filter((p) => isNum(p.quiz_score)).length;
    const metrics = `<div class="grid g4">
      ${metric("Learning sessions", n, "Conversations sent to EduLens")}
      ${metric("Topics studied", s.topics_studied_count, "Different topics you've covered")}
      ${metric("Average understanding", scoredU.length ? pct(s.average_understanding_score) : null, "Estimated from your conversations", "Not enough data yet")}
      ${metric("Quiz performance", isNum(s.average_actual_quiz_score) ? pct(s.average_actual_quiz_score) : null, quizCount ? `Across ${quizCount} graded quiz${quizCount > 1 ? "zes" : ""}` : "", "No quizzes yet")}
    </div>`;

    const chart = trend.length
      ? lineChart(trend.slice(-20), [{ key: "understanding_score", name: "Understanding" }, { key: "quiz_score", name: "Quiz score", dashed: true }], "Understanding and quiz scores across your recent sessions")
      : "";
    const chartCard = `<div class="card"><div class="card-head"><div><h2>Learning progress</h2><p>Your last ${Math.min(20, trend.length)} session${trend.length > 1 ? "s" : ""}. Gaps mean no score exists for that session.</p></div></div>${chart}</div>`;

    const bk = new Map((d.topic_wise_breakdown || []).map((b) => [norm(b.topic), b]));
    const improve = (d.areas_of_confusion || []).map((t) => bk.get(norm(t)) || { topic: t }).sort((a, b) => (a.latest_understanding === "confused" ? 0 : 1) - (b.latest_understanding === "confused" ? 0 : 1)).slice(0, 4);
    const analysed = (d.topic_wise_breakdown || []).some((b) => UNDERSTANDING[b.latest_understanding]);
    const improveCard = `<div class="card"><div class="card-head"><div><h2>Topics to improve</h2><p>Worth another look</p></div><a href="#/progress">See all</a></div>
      ${improve.length ? `<ul class="list">${improve.map((t) => `<li><a class="row-link li-main" href="${topicHref(t.topic)}"><div class="li-title">${esc(t.topic)}</div><div class="li-sub">${esc(t.subject || "")}${t.sessions ? ` · ${t.sessions} session${t.sessions > 1 ? "s" : ""}` : ""}</div></a>${understandingPill(t.latest_understanding)}</li>`).join("")}</ul>`
        : empty(analysed ? "Nothing needs attention" : "Not enough data yet", analysed ? "Your recent topics look solid. Keep going!" : "Topics will appear here once your sessions have been analysed.")}</div>`;

    const recent = (d.recent_sessions || []).slice(0, 4);
    const recentCard = `<div class="card"><div class="card-head"><div><h2>Recently studied</h2><p>Your latest sessions</p></div><a href="#/sessions">View all</a></div>
      <ul class="list">${recent.map((r) => `<li><a class="row-link li-main" href="#/sessions/${enc(r.session_id)}"><div class="li-title">${esc(r.topic || "Untitled topic")}</div><div class="li-sub">${esc(r.subject || "")} · ${esc(fmtDate(r.ingested_at))}</div></a>${understandingPill(r.understanding_class)}</li>`).join("")}</ul></div>`;

    const dist = d.understanding_distribution || {};
    const first = improve[0];
    const summary = `<div class="card"><h2>Summary</h2><p style="margin-top:8px">You've sent <strong>${n}</strong> session${n > 1 ? "s" : ""} covering <strong>${s.topics_studied_count}</strong> topic${s.topics_studied_count > 1 ? "s" : ""}. ${dist.understood || 0} showed good understanding, ${dist.partial || 0} partial, and ${dist.confused || 0} need${(dist.confused || 0) === 1 ? "s" : ""} review. ${first ? `A good next step: revisit <a href="${topicHref(first.topic)}">${esc(first.topic)}</a>.` : ""}${quizCount === 0 ? " Record a quiz result to see how your understanding holds up in practice." : ""}</p></div>`;

    paint(token, head + `<div class="stack">${metrics}${chartCard}<div class="grid g2">${improveCard}${recentCard}</div>${summary}</div>`, { title: "Overview" });
  }

  /* ================= My Sessions ================= */
  async function viewSessions(token) {
    const perf = await getS("/student-performance");
    const rows = perf.topic_wise_breakdown || [];
    const subjects = Array.from(new Set(rows.map((r) => r.subject).filter(Boolean))).sort();
    const topics = Array.from(new Set(rows.map((r) => r.topic).filter(Boolean))).sort();
    const f = state.sessionFilters;
    const opt = (list, cur, all) => `<option value="">${all}</option>` + list.map((v) => `<option value="${esc(v)}" ${v === cur ? "selected" : ""}>${esc(v)}</option>`).join("");
    const html = pageHead("My Sessions", "Every learning conversation you've sent to EduLens.") + `
      <div class="toolbar" role="search">
        <label class="grow"><span class="sr-only">Search sessions</span><input type="search" data-filter="search" placeholder="Search by topic or subject" value="${esc(f.search)}" /></label>
        <label><span class="sr-only">Subject</span><select data-filter="subject">${opt(subjects, f.subject, "All subjects")}</select></label>
        <label><span class="sr-only">Topic</span><select data-filter="topic">${opt(topics, f.topic, "All topics")}</select></label>
        <label><span class="sr-only">Understanding</span><select data-filter="cls"><option value="">Any understanding</option>${Object.entries(UNDERSTANDING).map(([k, v]) => `<option value="${k}" ${k === f.cls ? "selected" : ""}>${v.label}</option>`).join("")}</select></label>
      </div>
      <div class="card flush" id="session-list" aria-live="polite">${`<div class="skel" style="height:140px;margin:20px"></div>`}</div>`;
    if (!paint(token, html, { title: "My Sessions" })) return;
    loadSessionList(token);
  }
  async function loadSessionList(token) {
    const f = state.sessionFilters, box = $("#session-list");
    if (!box) return;
    try {
      const r = await getS("/sessions", { search: f.search.trim(), subject: f.subject, topic: f.topic, class_filter: f.cls, limit: f.limit, offset: f.page * f.limit });
      if (token !== state.token || !$("#session-list")) return;
      if (!r.sessions.length) {
        const filtered = f.search || f.subject || f.topic || f.cls;
        box.innerHTML = filtered ? empty("No sessions match", "Try a different search or clear a filter.", `<button class="btn" data-action="clear-filters">Clear filters</button>`) : empty("No sessions yet", "Send your first study conversation to see it here.", ingestBtn());
        return;
      }
      const from = f.page * f.limit + 1, to = f.page * f.limit + r.sessions.length;
      box.innerHTML = r.sessions.map((s) => `
        <div class="session-row">
          <div class="li-main"><div class="li-title">${esc(s.topic || "Untitled topic")}</div><div class="li-sub">${esc(s.subject || "No subject")}</div></div>
          <div class="li-sub">${esc(fmtDate(s.ingested_at))} · ${esc(agentLabel(s.source_agent))}</div>
          <div>${understandingPill(s.understanding_class)}</div>
          <div class="quiz-cell">${isNum(s.actual_quiz_score) ? `Quiz <strong>${pct(s.actual_quiz_score)}</strong>` : `<span class="muted">Awaiting quiz</span>`}</div>
          <div><a class="btn btn-sm" href="#/sessions/${enc(s.session_id)}" aria-label="View details for ${esc(s.topic || "session")}">View details</a></div>
        </div>`).join("") + `<div class="pager"><span class="small muted">Showing ${from}–${to} of ${r.total_count}</span><span><button class="btn btn-sm" data-action="page-prev" ${f.page === 0 ? "disabled" : ""}>Previous</button> <button class="btn btn-sm" data-action="page-next" ${to >= r.total_count ? "disabled" : ""}>Next</button></span></div>`;
    } catch (e) { if (token === state.token && $("#session-list")) $("#session-list").innerHTML = `<div style="padding:20px">${errorBox(e)}</div>`; }
  }

  /* ================= Session detail ================= */
  async function viewSessionDetail(token, id) {
    let d;
    try { d = await getS(`/sessions/${enc(id)}`); }
    catch (e) { return paint(token, `<a class="crumb" href="#/sessions">${icon("back")} My Sessions</a>${errorBox(e)}`); }
    const m = d.summary_metrics, a = d.analytics, t0 = a?.topics?.[0], q = d.linked_quiz;
    const cls = m?.predicted_class;
    const quizBlock = isNum(m?.actual_quiz_score) || q
      ? metric("Quiz result", pct(m?.actual_quiz_score ?? q?.actual_score_percent), q ? `${q.correct_count} of ${q.total_questions} correct` : "Graded quiz")
      : `<div class="card metric"><span class="label">Quiz result</span><span class="value is-empty">Quiz not attempted</span><span class="note"><button class="btn btn-sm" data-action="open-quiz" data-session="${esc(id)}">Record quiz result</button></span></div>`;
    const conf = m?.expressed_confidence_level ? `${m.expressed_confidence_level.charAt(0).toUpperCase()}${m.expressed_confidence_level.slice(1)}` : null;
    const metrics = `<div class="grid g4">
      <div class="card metric"><span class="label">Topic understanding</span><span class="value" style="font-size:20px;padding:3px 0">${m ? understandingPill(cls) : "<span class='muted'>Not available</span>"}</span><span class="note">${isNum(m?.understanding_score) ? `Predicted at ${pct(m.understanding_score)} from the conversation` : "Predicted from the conversation"}</span></div>
      ${quizBlock}
      ${metric("Confidence in your wording", conf, `${isNum(m?.expressed_confidence_score) ? `Score ${Math.round(m.expressed_confidence_score)}/100. ` : ""}Based on how you phrased things — not a personality measure`, "Not available")}
      ${metric("Overall topic score", pct(m?.overall_performance_score), "Combines understanding, quiz and confidence", "Available after a quiz")}
    </div>`;

    const transcript = (d.messages || []).length
      ? `<div class="transcript" tabindex="0" aria-label="Conversation transcript">${d.messages.map((x) => { const stu = ["user", "student", "human"].includes(norm(x.role)); return `<div class="msg ${stu ? "student" : "ai"}"><span class="who-tag">${stu ? "You" : "AI"}</span>${esc(x.content)}</div>`; }).join("")}</div>`
      : empty("No transcript saved", "The conversation text isn't available for this session.");

    const cs = t0?.conversational_signals || {}, sg = t0?.expressed_confidence?.signals || {};
    const sigs = [["Messages you sent", sg.total_student_turns], ["Questions you asked", cs.question_count], ["Signs of understanding", cs.understanding_markers], ["Signs of confusion", cs.confusion_markers], ["Asked for clarification", cs.clarification_markers], ["Hesitant phrases", sg.hesitation_count], ["Uncertain phrases", sg.uncertainty_count], ["Explained in your own words", sg.independent_explanations]].filter(([, v]) => isNum(v));
    const signals = sigs.length ? `<div class="sig-list">${sigs.map(([k, v]) => `<div><span>${k}</span><strong>${Math.round(v)}</strong></div>`).join("")}</div>` : `<p class="muted">Not enough data yet.</p>`;

    const quizCard = q ? `<div class="card"><div class="card-head"><h2>Quiz answers</h2><a href="#/quizzes">All quizzes</a></div>${quizItems(q.question_breakdown)}</div>` : "";
    const next = m?.pedagogical_feedback ? `<div class="card"><h2>Suggested next step</h2><p style="margin-top:8px">${esc(m.pedagogical_feedback)}</p>${m.discrepancy_detected && m.discrepancy_details ? `<p class="notice" style="margin-top:12px">${esc(typeof m.discrepancy_details === "string" ? m.discrepancy_details : JSON.stringify(m.discrepancy_details))}</p>` : ""}</div>` : "";

    const probs = t0?.understanding?.probabilities;
    const advanced = `<details class="disclosure"><summary>Advanced details</summary><div class="body"><dl class="kv">
      <dt>Session ID</dt><dd>${esc(d.session_id)}</dd>
      <dt>Status</dt><dd>${esc(d.status || "—")}</dd>
      ${t0?.understanding ? `<dt>Model used</dt><dd>${esc(t0.understanding.model_used || "—")}</dd><dt>Model certainty</dt><dd>${isNum(t0.understanding.confidence) ? pct(t0.understanding.confidence * 100) : "—"}</dd>` : ""}
      ${probs ? `<dt>Class probabilities</dt><dd>${Object.entries(probs).map(([k, v]) => `${esc(UNDERSTANDING[k]?.label || k)}: ${pct(v * 100)}`).join(" · ")}</dd>` : ""}
      ${isNum(t0?.predicted_quiz_performance?.score) ? `<dt>Estimated quiz score</dt><dd>${pct(t0.predicted_quiz_performance.score)} — a model estimate, not a real quiz result</dd>` : ""}
      ${t0?.overall_performance?.formula ? `<dt>Overall score formula</dt><dd>${esc(t0.overall_performance.formula)}</dd>` : ""}
    </dl>${t0?.feature_explanation ? `<h3 style="margin-top:16px">What influenced the prediction</h3><pre class="code">${esc(JSON.stringify(t0.feature_explanation, null, 2).slice(0, 2500))}</pre>` : ""}</div></details>`;

    paint(token, `<a class="crumb" href="#/sessions">${icon("back")} My Sessions</a>
      ${pageHead(d.topic || "Untitled topic", `${d.subject || "No subject"} · ${fmtDate(d.ingested_at)} · ${agentLabel(d.source_agent)} · ${d.turns_count ?? "?"} messages`, `<button class="btn" data-action="copy" data-copy="${esc(d.session_id)}">${icon("copy")} Copy session ID</button>`)}
      <div class="stack">${metrics}${next}<div class="card"><div class="card-head"><div><h2>Conversation</h2><p>What you and the AI said</p></div></div>${transcript}</div>
      <div class="card"><div class="card-head"><div><h2>What the conversation showed</h2><p>Counts of learning signals found in your messages</p></div></div>${signals}</div>${quizCard}${advanced}</div>`, { title: d.topic || "Session" });
  }
  function quizItems(items) {
    return (items || []).map((b) => `<div class="q-item"><div style="font-weight:600">${esc(b.question_text || b.question_id)}</div>
      <div class="small">${b.is_correct ? pill("Correct", "green") : b.status === "unanswered" ? pill("Not answered", "amber") : pill("Incorrect", "red")}
      <span class="muted">Your answer:</span> ${esc(b.submitted_answer || "—")}${b.is_correct ? "" : ` <span class="muted">· Correct answer:</span> ${esc(b.correct_answer ?? "—")}`}</div></div>`).join("") || `<p class="muted">No question details saved.</p>`;
  }

  /* ================= My Progress ================= */
  async function viewProgress(token) {
    const [perf, cat] = await Promise.all([getS("/student-performance"), api("/topics").catch(() => null)]);
    const progress = perf.learning_progress || [];
    if (!perf.total_sessions) {
      return paint(token, pageHead("My Progress", "Your understanding and quiz results over time.") + `<div class="card">${empty("No progress to show yet", "Once you've sent a study session, your topics and trends will appear here.", ingestBtn())}</div>`, { title: "My Progress" });
    }
    const tab = state.progressTab;
    const tabs = `<div class="tabs" role="tablist" aria-label="Progress views"><button class="tab" role="tab" data-action="tab" data-tab="overall" aria-selected="${tab === "overall"}">Overall progress</button><button class="tab" role="tab" data-action="tab" data-tab="topics" aria-selected="${tab === "topics"}">Topic progress</button></div>`;
    let body;
    if (tab === "overall") {
      const dist = perf.understanding_distribution || {};
      const total = (dist.understood || 0) + (dist.partial || 0) + (dist.confused || 0);
      const hasU = progress.some((p) => isNum(p.understanding_score)), hasQ = progress.some((p) => isNum(p.quiz_score));
      const rows = topicRows(progress, perf.topic_wise_breakdown);
      const needs = rows.filter((r) => (perf.areas_of_confusion || []).some((t) => norm(t) === norm(r.topic)));
      body = `<div class="stack"><div class="grid g4" style="grid-template-columns:repeat(3,1fr)">
        ${metric("Average understanding", hasU ? pct(perf.average_understanding_score) : null, "Estimated from your conversations", "Not enough data yet")}
        ${metric("Average quiz score", pct(perf.average_actual_quiz_score), hasQ ? "From graded quizzes only" : "", "No quizzes yet")}
        ${metric("Overall topic performance", pct(perf.average_overall_score), "Needs a graded quiz for each session", "Available after a quiz")}
      </div>
      <div class="grid g2">
        <div class="card"><div class="card-head"><div><h2>Understanding trend</h2><p>Estimated from each conversation</p></div></div>${hasU ? lineChart(progress.slice(-30), [{ key: "understanding_score", name: "Understanding" }], "Understanding trend across sessions") : empty("Not enough data yet", "Scores appear once sessions are analysed.")}</div>
        <div class="card"><div class="card-head"><div><h2>Quiz performance trend</h2><p>Graded quizzes only</p></div></div>${hasQ ? lineChart(progress.slice(-30), [{ key: "quiz_score", name: "Quiz score" }], "Quiz score trend across sessions") : empty("Quiz not attempted", "Record a quiz result for a session and its score will appear here.")}</div>
      </div>
      <div class="grid g2">
        <div class="card"><div class="card-head"><div><h2>How your sessions were rated</h2></div></div>${total ? ["understood", "partial", "confused"].map((k) => `<div style="margin-bottom:12px"><div class="small" style="display:flex;justify-content:space-between"><span>${UNDERSTANDING[k].label}</span><strong>${dist[k] || 0} of ${total}</strong></div><div class="progress" aria-hidden="true"><span style="width:${((dist[k] || 0) / total) * 100}%;background:var(--${UNDERSTANDING[k].tone})"></span></div></div>`).join("") : `<p class="muted">Not enough data yet.</p>`}</div>
        <div class="card"><div class="card-head"><div><h2>Topics requiring improvement</h2></div></div>${needs.length ? `<ul class="list">${needs.map((t) => `<li><a class="row-link li-main" href="${topicHref(t.topic)}"><div class="li-title">${esc(t.topic)}</div><div class="li-sub">${esc(t.subject || "")}</div></a>${understandingPill(t.latest)}</li>`).join("")}</ul>` : `<p class="muted">Nothing needs attention right now.</p>`}</div>
      </div></div>`;
    } else {
      const rows = topicRows(progress, perf.topic_wise_breakdown).sort((a, b) => a.topic.localeCompare(b.topic));
      const studied = new Set(rows.map((r) => norm(r.topic)));
      const unstudied = (cat?.topics || []).filter((c) => !studied.has(norm(c.topic)));
      const bySubject = unstudied.reduce((m, c) => ((m[c.subject] = m[c.subject] || []).push(c.topic), m), {});
      body = `<div class="stack"><div class="card flush"><div style="padding:20px 20px 4px"><h2>Topics you've studied</h2><p class="small muted">Only topics from your own sessions are listed here.</p></div><div class="table-wrap"><table><thead><tr><th>Topic</th><th>Sessions</th><th>Understanding</th><th>Quiz</th><th>Status</th></tr></thead><tbody>
        ${rows.map((r) => `<tr><td><a href="${topicHref(r.topic)}"><strong>${esc(r.topic)}</strong></a><div class="li-sub">${esc(r.subject || "")}</div></td><td>${r.sessions}</td><td>${UNDERSTANDING[r.latest] ? esc(UNDERSTANDING[r.latest].label) : "Not analysed yet"}${isNum(r.avgU) ? `<div class="li-sub">Average ${pct(r.avgU)}</div>` : ""}</td><td>${isNum(r.avgQ) ? `${pct(r.avgQ)}<div class="li-sub">${r.quizCount} quiz${r.quizCount > 1 ? "zes" : ""}</div>` : `<span class="muted">Quiz not attempted</span>`}</td><td>${progressPill(r.latest)}</td></tr>`).join("")}</tbody></table></div></div>
        ${unstudied.length ? `<details class="disclosure"><summary>Not studied yet — ${unstudied.length} topics in the course catalogue</summary><div class="body"><p class="small muted" style="margin-bottom:12px">These are available in the curriculum but are not part of your progress.</p>${Object.entries(bySubject).sort().map(([s, ts]) => `<h3 style="margin:14px 0 8px">${esc(s)}</h3><div class="chips">${ts.map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</div>`).join("")}</div></details>` : ""}</div>`;
    }
    paint(token, pageHead("My Progress", "Your understanding and quiz results over time.") + tabs + body, { title: "My Progress", focus: false });
    window.scrollTo(0, 0);
  }

  async function viewTopicDetail(token, topic) {
    const [sum, perf] = await Promise.all([getS("/topic-summary", { topic }), getS("/student-performance")]);
    const crumb = `<a class="crumb" href="#/progress">${icon("back")} My Progress</a>`;
    const mine = (perf.learning_progress || []).filter((p) => norm(p.topic) === norm(topic));
    if (!sum.total_sessions && !mine.length) return paint(token, crumb + pageHead(topic) + `<div class="card">${empty("Not studied", "You haven't sent any sessions for this topic yet.", ingestBtn())}</div>`, { title: topic });
    const hist = sum.historical_trend || mine;
    const hasU = hist.some((p) => isNum(p.understanding_score)), hasQ = hist.some((p) => isNum(p.quiz_score));
    const latest = mine[mine.length - 1]?.understanding_class;
    const points = hist.map((h) => ({ ...h, topic }));
    paint(token, crumb + pageHead(sum.topic || topic, `${sum.subject || ""} · ${sum.total_sessions} session${sum.total_sessions > 1 ? "s" : ""}`, progressPill(latest)) + `<div class="stack">
      <div class="grid g4" style="grid-template-columns:repeat(3,1fr)">
        ${metric("Average understanding", hasU ? pct(sum.average_understanding_score) : null, "Estimated from conversations", "Not enough data yet")}
        ${metric("Quiz performance", pct(sum.quiz_performance?.average_score_percent), sum.quiz_performance?.quizzes_taken ? `${sum.quiz_performance.quizzes_taken} graded quiz${sum.quiz_performance.quizzes_taken > 1 ? "zes" : ""}` : "", "Quiz not attempted")}
        ${metric("Overall topic performance", pct(sum.overall_topic_performance?.composite_score), "", "Available after a quiz")}
      </div>
      <div class="card"><div class="card-head"><div><h2>History</h2><p>Gaps mean no score exists for that session</p></div></div>${hasU || hasQ ? lineChart(points, [{ key: "understanding_score", name: "Understanding" }, { key: "quiz_score", name: "Quiz score", dashed: true }], `History for ${topic}`) : empty("Not enough data yet", "Scores appear once sessions are analysed.")}</div>
      <div class="card"><div class="card-head"><h2>Sessions on this topic</h2></div><ul class="list">${[...mine].reverse().map((p) => `<li><a class="row-link li-main" href="#/sessions/${enc(p.session_id)}"><div class="li-title">${esc(fmtDate(p.date))}</div><div class="li-sub">${isNum(p.quiz_score) ? `Quiz ${pct(p.quiz_score)}` : "Quiz not attempted"}</div></a>${understandingPill(p.understanding_class)}</li>`).join("")}</ul></div></div>`, { title: topic });
  }

  /* ================= Quizzes ================= */
  async function viewQuizzes(token) {
    const [qz, ss] = await Promise.all([getS("/quizzes", { limit: 100 }), getS("/sessions", { limit: 200 })]);
    const quizzes = qz.quizzes || [];
    const pending = (ss.sessions || []).filter((s) => !isNum(s.actual_quiz_score) && !quizzes.some((q) => q.session_id === s.session_id));
    const head = pageHead("Quizzes", "Quiz results show how well you actually know a topic.", `<button class="btn btn-primary" data-action="open-quiz">${icon("plus")} Record quiz result</button>`);
    const metrics = `<div class="grid g4" style="grid-template-columns:repeat(3,1fr)">${metric("Quizzes completed", qz.total_count, "")}${metric("Average score", pct(avg(quizzes.map((q) => q.actual_quiz_score ?? q.actual_score_percent))), quizzes.length ? "Across graded quizzes" : "", "No quizzes yet")}${metric("Awaiting a quiz", pending.length, "Sessions with no quiz result yet")}</div>`;
    const pendingCard = pending.length ? `<div class="card"><div class="card-head"><div><h2>Awaiting quiz</h2><p>These sessions have no quiz result, so no score is shown</p></div></div><ul class="list">${pending.slice(0, 8).map((s) => `<li><a class="row-link li-main" href="#/sessions/${enc(s.session_id)}"><div class="li-title">${esc(s.topic || "Untitled topic")}</div><div class="li-sub">${esc(s.subject || "")} · ${esc(fmtDate(s.ingested_at))}</div></a><button class="btn btn-sm" data-action="open-quiz" data-session="${esc(s.session_id)}">Record result</button></li>`).join("")}</ul>${pending.length > 8 ? `<p class="small muted" style="margin-top:10px">and ${pending.length - 8} more in <a href="#/sessions">My Sessions</a></p>` : ""}</div>` : "";
    const history = quizzes.length ? `<div class="card flush"><div style="padding:20px 20px 4px"><h2>Quiz history</h2></div><div class="table-wrap"><table><thead><tr><th>Topic</th><th>Date</th><th>Questions</th><th>Score</th><th>Status</th><th><span class="sr-only">Answers</span></th></tr></thead><tbody>
      ${quizzes.map((q) => `<tr><td><strong>${esc(q.topic || "General quiz")}</strong><div class="li-sub">${esc(q.subject || "")}</div></td><td>${esc(fmtDate(q.evaluated_at))}</td><td>${q.total_questions}</td><td><strong>${pct(q.actual_score_percent)}</strong><div class="li-sub">${q.correct_count} correct</div></td><td>${q.unanswered_count ? pill(`${q.unanswered_count} unanswered`, "amber") : pill("Completed", "green")}</td><td><button class="btn btn-sm" data-action="toggle-quiz" data-id="${esc(q.quiz_id)}" aria-expanded="false">View answers</button></td></tr><tr hidden id="qd-${esc(q.quiz_id)}"><td colspan="6"></td></tr>`).join("")}</tbody></table></div></div>`
      : `<div class="card">${empty("No quizzes yet", "After studying a topic, take a short quiz elsewhere and record your answers here. EduLens grades them against your answer key.", `<button class="btn btn-primary" data-action="open-quiz">Record quiz result</button>`)}</div>`;
    paint(token, head + `<div class="stack">${metrics}${pendingCard}${history}</div>`, { title: "Quizzes" });
  }

  /* ================= Connect & Ingest ================= */
  async function viewConnect(token) {
    const origin = location.origin;
    const stdio = JSON.stringify({ mcpServers: { edulens: { command: "<path-to-your-venv>/python", args: ["-m", "backend.mcp_server.server", "--transport", "stdio"], env: { EDULENS_AUTH_ENABLED: "true", EDULENS_DEV_TOKEN: "<your-token>" } } } }, null, 2);
    const http = JSON.stringify({ mcpServers: { "edulens-remote": { url: "http://127.0.0.1:8001/mcp", headers: { Authorization: "Bearer <your-token>" } } } }, null, 2);
    const curl = `curl -X POST ${origin}/api/ingest \\\n  -H "Content-Type: application/json" \\\n  -d '{"student_id":"${state.studentId}","subject":"DAA","topic":"Binary Search","source_agent":"claude","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}'`;
    const codeBlock = (label, code) => `<div style="margin-top:12px"><div style="display:flex;justify-content:space-between;align-items:center"><strong class="small">${label}</strong><button class="btn btn-sm btn-ghost" data-action="copy" data-copy="${esc(code)}">${icon("copy")} Copy</button></div><pre class="code">${esc(code)}</pre></div>`;
    paint(token, pageHead("Connect & Ingest", "EduLens doesn't teach you — you keep learning with the AI tool you like, and EduLens measures how it's going.") + `<div class="stack">
      <div class="card"><div class="steps">
        <div class="step"><span class="step-num">1</span><h3>Learn with your AI tool</h3><p class="muted small">Study a topic with ChatGPT, Claude, Gemini or any other assistant.</p></div>
        <div class="step"><span class="step-num">2</span><h3>Send the conversation</h3><p class="muted small">Paste it here, or send it through one of the connections below.</p></div>
        <div class="step"><span class="step-num">3</span><h3>See your progress</h3><p class="muted small">EduLens estimates your understanding and shows what to review next.</p></div>
      </div></div>
      <div class="card"><h2>Paste or upload a conversation</h2><p class="muted" style="margin:6px 0 14px">The quickest way. Works with any AI tool, no setup needed.</p>${ingestBtn()}</div>
      <details class="disclosure"><summary>Claude Desktop (MCP)</summary><div class="body"><p class="muted small">Lets Claude send sessions to EduLens for you. The local connection (stdio) has been tested on a local machine; the remote HTTP option needs the MCP server running (<code>python -m backend.mcp_server.server --transport streamable --port 8001</code>). Replace the placeholders with your own values.</p>${codeBlock("Local (stdio)", stdio)}${codeBlock("Remote (Streamable HTTP)", http)}</div></details>
      <details class="disclosure"><summary>Send sessions from your own code (REST API)</summary><div class="body"><p class="muted small">Post a conversation to the ingest endpoint. Use roles <code>user</code> for you and <code>assistant</code> for the AI.</p>${codeBlock("Example request", curl)}</div></details>
      <div class="notice"><strong>ChatGPT:</strong> a live ChatGPT custom MCP connection isn't supported yet, because it needs OAuth sign-in that EduLens doesn't have. Paste a ChatGPT conversation above instead.</div>
    </div>`, { title: "Connect & Ingest" });
  }

  /* ================= Settings ================= */
  async function viewSettings(token) {
    const pref = localStorage.getItem("edulens_theme") || "system";
    const [health, ss] = await Promise.all([api("/health").then(() => true).catch(() => false), getS("/sessions", { limit: 200 }).catch(() => null)]);
    const counts = {}; (ss?.sessions || []).forEach((s) => { const k = agentLabel(s.source_agent); counts[k] = (counts[k] || 0) + 1; });
    paint(token, pageHead("Settings", "Keep it simple: who you are, how it looks, and how EduLens connects.") + `<div class="stack">
      <div class="card"><h2>Student profile</h2><p class="muted small" style="margin:4px 0 14px">Choose whose learning data you're viewing.</p>
        <div class="grid g2"><label class="field">Student ID<select id="set-student" style="width:100%">${state.students.concat(state.students.includes(state.studentId) ? [] : [state.studentId]).map((id) => `<option ${id === state.studentId ? "selected" : ""}>${esc(id)}</option>`).join("")}</select></label>
        <label class="field">Switch to another ID<span style="display:flex;gap:8px"><input type="text" id="set-new-student" placeholder="e.g. student_002" /><button class="btn" data-action="apply-student">Switch</button></span></label></div></div>
      <div class="card"><h2>Appearance</h2><div class="tabs" role="radiogroup" aria-label="Theme" style="margin:12px 0 0">${["light", "dark", "system"].map((t) => `<button class="tab" role="radio" aria-checked="${pref === t}" aria-selected="${pref === t}" data-action="theme" data-theme="${t}">${t.charAt(0).toUpperCase() + t.slice(1)}</button>`).join("")}</div></div>
      <div class="card"><h2>Connection</h2><dl class="kv" style="margin-top:12px"><dt>EduLens server</dt><dd>${health ? pill("Reachable", "green") : pill("Not reachable", "red")}</dd>
        <dt>Sessions received from</dt><dd>${Object.keys(counts).length ? Object.entries(counts).map(([k, v]) => `${esc(k)} (${v})`).join(", ") : "No sessions yet"}</dd></dl>
        <p class="small muted" style="margin-top:12px">EduLens can't tell from this page whether an AI tool is currently connected through MCP; it only shows which tools have already sent sessions. See <a href="#/connect">Connect &amp; Ingest</a> for setup.</p></div>
      <div class="card"><h2>Data &amp; privacy</h2><ul style="margin:10px 0 0;padding-left:20px;color:var(--text-2)"><li>EduLens only analyses conversations you choose to send.</li><li>Your sessions are kept separate from other students' data.</li><li>This page never shows passwords, tokens or server settings.</li></ul></div>
    </div>`, { title: "Settings" });
  }

  /* ================= Transcript parsing ================= */
  const STUDENT_ROLES = ["you", "me", "student", "user", "human", "learner"];
  const AI_ROLES = ["ai", "assistant", "chatgpt", "claude", "gemini", "tutor", "teacher", "bot", "gpt"];
  function parseTranscript(raw) {
    const text = raw.trim();
    if (/^[\[{]/.test(text)) {
      try {
        const j = JSON.parse(text);
        const list = Array.isArray(j) ? j : j.messages || null;
        if (list) return list.map((m) => (typeof m === "string" ? { role: "user", content: m } : { role: m.role || "user", content: String(m.content ?? m.text ?? "") })).filter((m) => m.content.trim());
        if (Array.isArray(j.turns)) return j.turns.flatMap((t) => [t.student_message && { role: "user", content: t.student_message }, t.teacher_message && { role: "assistant", content: t.teacher_message }]).filter(Boolean);
      } catch { /* not JSON — treat as plain text */ }
    }
    const re = new RegExp(`^\\s*(${[...STUDENT_ROLES, ...AI_ROLES].join("|")})\\s*[:：]\\s*(.*)$`, "i");
    const lines = text.split(/\r?\n/);
    if (!lines.some((l) => re.test(l))) return lines.map((l) => l.trim()).filter(Boolean).map((content) => ({ role: "user", content }));
    const out = [];
    lines.forEach((l) => {
      const m = l.match(re);
      if (m) out.push({ role: STUDENT_ROLES.includes(m[1].toLowerCase()) ? "user" : "assistant", content: m[2] });
      else if (out.length) out[out.length - 1].content += "\n" + l;
      else if (l.trim()) out.push({ role: "user", content: l.trim() });
    });
    return out.map((m) => ({ ...m, content: m.content.trim() })).filter((m) => m.content);
  }

  /* ================= Dialogs ================= */
  function openIngest() {
    const dlg = $("#ingest-dialog");
    dlg.innerHTML = `<form method="dialog" id="ingest-form" novalidate>
      <div class="dlg-head"><h2 id="ingest-title">Ingest a study session</h2><button type="button" class="btn btn-ghost icon-btn" data-action="close-dialog" aria-label="Close">${icon("x")}</button></div>
      <div class="dlg-body">
        <div class="grid g2"><label class="field">Subject<input type="text" id="ing-subject" placeholder="e.g. Algorithms" required /></label>
        <label class="field">Topic<input type="text" id="ing-topic" placeholder="e.g. Binary Search" required /></label></div>
        <label class="field">AI tool you used<select id="ing-agent" style="width:100%"><option value="chatgpt">ChatGPT</option><option value="claude">Claude</option><option value="gemini">Gemini</option><option value="cursor">Cursor</option><option value="generic">Other</option></select></label>
        <label class="field">Conversation<textarea id="ing-text" placeholder="You: Can you explain binary search?&#10;AI: Sure! ..." required></textarea>
        <span class="field-hint">Start lines with “You:” and “AI:” so EduLens can tell who said what. Without labels, every line is treated as your message. JSON exports also work.</span></label>
        <label class="field">…or upload a file<input type="file" id="ing-file" accept=".txt,.md,.json,text/plain,application/json" /></label>
        <p class="small" id="ing-error" role="alert" style="color:var(--red)" hidden></p>
      </div>
      <div class="dlg-foot"><span class="status" id="ing-status"></span><button type="button" class="btn" data-action="close-dialog">Cancel</button><button type="submit" class="btn btn-primary" id="ing-submit">Analyse session</button></div></form>`;
    dlg.showModal();
    $("#ing-subject").focus();
    $("#ing-file").addEventListener("change", (e) => {
      const f = e.target.files[0]; if (!f) return;
      if (f.size > 2_000_000) { showIngErr("That file is too large (limit 2 MB)."); return; }
      f.text().then((t) => { $("#ing-text").value = t; toast(`Loaded ${f.name}`); }).catch(() => showIngErr("Couldn't read that file."));
    });
    $("#ingest-form").addEventListener("submit", submitIngest);
  }
  function showIngErr(msg) { const el = $("#ing-error"); if (el) { el.textContent = msg; el.hidden = !msg; } }
  async function submitIngest(e) {
    e.preventDefault(); showIngErr("");
    const subject = $("#ing-subject").value.trim(), topic = $("#ing-topic").value.trim(), raw = $("#ing-text").value;
    if (!subject || !topic) return showIngErr("Please enter both a subject and a topic.");
    if (!raw.trim()) return showIngErr("Paste a conversation or upload a file first.");
    const messages = parseTranscript(raw);
    if (!messages.some((m) => m.role === "user" && m.content.trim())) return showIngErr("EduLens couldn't find any messages from you in that text.");
    const btn = $("#ing-submit"); btn.disabled = true; $("#ing-status").textContent = "Analysing your conversation…";
    try {
      const res = await api("/ingest", { method: "POST", body: { student_id: state.studentId, subject, topic, source_agent: $("#ing-agent").value, messages } });
      $("#ingest-dialog").close();
      toast("Session added. Here's what EduLens found.");
      if (!state.students.includes(state.studentId)) { state.students.push(state.studentId); renderIdentity(); }
      location.hash = `#/sessions/${enc(res.session_id)}`;
      if (location.hash === `#/sessions/${enc(res.session_id)}`) router();
    } catch (err) { btn.disabled = false; $("#ing-status").textContent = ""; showIngErr(err.message); toast("Couldn't add the session", "error"); }
  }

  async function openQuiz(sessionId = "") {
    const dlg = $("#quiz-dialog");
    dlg.innerHTML = `<div class="dlg-body"><div class="skel" style="height:160px"></div></div>`;
    dlg.showModal();
    let sessions = [];
    try { sessions = (await getS("/sessions", { limit: 200 })).sessions || []; } catch { /* the form still works without a session link */ }
    const sel = sessions.find((s) => s.session_id === sessionId);
    dlg.innerHTML = `<form method="dialog" id="quiz-form" novalidate>
      <div class="dlg-head"><h2 id="quiz-title">Record quiz result</h2><button type="button" class="btn btn-ghost icon-btn" data-action="close-dialog" aria-label="Close">${icon("x")}</button></div>
      <div class="dlg-body">
        <label class="field">Session this quiz belongs to<select id="qz-session" style="width:100%"><option value="">Not linked to a session</option>${sessions.map((s) => `<option value="${esc(s.session_id)}" data-topic="${esc(s.topic || "")}" data-subject="${esc(s.subject || "")}" ${s.session_id === sessionId ? "selected" : ""}>${esc(s.topic || "Untitled")} — ${esc(fmtDate(s.ingested_at))}${isNum(s.actual_quiz_score) ? " (already has a quiz)" : ""}</option>`).join("")}</select></label>
        <div class="grid g2"><label class="field">Subject<input type="text" id="qz-subject" value="${esc(sel?.subject || "")}" /></label><label class="field">Topic<input type="text" id="qz-topic" value="${esc(sel?.topic || "")}" /></label></div>
        <div><strong class="small">Questions</strong><p class="field-hint" style="margin-bottom:8px">Enter each question, the correct answer, and what you answered. Answers are compared ignoring capitalisation. Leave “Your answer” empty if you skipped it.</p><div id="qz-rows"></div><button type="button" class="btn btn-sm" data-action="add-q">${icon("plus")} Add question</button></div>
        <p class="small" id="qz-error" role="alert" style="color:var(--red)" hidden></p>
      </div>
      <div class="dlg-foot"><span class="status" id="qz-status"></span><button type="button" class="btn" data-action="close-dialog">Cancel</button><button type="submit" class="btn btn-primary" id="qz-submit">Grade quiz</button></div></form>`;
    addQuestionRow(); addQuestionRow();
    $("#qz-session").addEventListener("change", (e) => { const o = e.target.selectedOptions[0]; if (o.value) { $("#qz-topic").value = o.dataset.topic; $("#qz-subject").value = o.dataset.subject; } });
    $("#quiz-form").addEventListener("submit", submitQuiz);
    $("#qz-session").focus();
  }
  function addQuestionRow() {
    const wrap = $("#qz-rows"); if (!wrap) return;
    const i = wrap.children.length + 1;
    wrap.insertAdjacentHTML("beforeend", `<div class="q-row"><label class="field">Question ${i}<input type="text" class="q-text" /></label><label class="field">Correct answer<input type="text" class="q-correct" /></label><label class="field">Your answer<input type="text" class="q-given" /></label><button type="button" class="btn btn-ghost icon-btn" data-action="remove-q" aria-label="Remove question ${i}">${icon("x")}</button></div>`);
  }
  async function submitQuiz(e) {
    e.preventDefault();
    const err = (m) => { const el = $("#qz-error"); el.textContent = m; el.hidden = !m; };
    err("");
    const qs = [], key = {}, given = {};
    $$("#qz-rows .q-row").forEach((row) => {
      const text = $(".q-text", row).value.trim(), correct = $(".q-correct", row).value.trim(), ans = $(".q-given", row).value.trim();
      if (!text && !correct && !ans) return;
      const id = `q${qs.length + 1}`;
      qs.push({ text, correct, ans, id }); key[id] = correct; given[id] = ans;
    });
    if (!qs.length) return err("Add at least one question.");
    if (qs.some((q) => !q.text || !q.correct)) return err("Each question needs question text and a correct answer.");
    const sessionId = $("#qz-session").value, topic = $("#qz-topic").value.trim(), subject = $("#qz-subject").value.trim();
    if (!topic && !sessionId) return err("Enter a topic or link a session.");
    const btn = $("#qz-submit"); btn.disabled = true; $("#qz-status").textContent = "Grading…";
    try {
      const res = await api("/quizzes", { method: "POST", body: { student_id: state.studentId, session_id: sessionId || null, subject: subject || null, topic: topic || null,
        questions: qs.map((q) => ({ question_id: q.id, question_text: q.text, topic: topic || null, student_answer: q.ans, correct_answer: q.correct })), answer_key: key, student_answers: given } });
      $("#quiz-dialog").close();
      toast(`Quiz graded: ${pct(res.actual_score_percent)} (${res.correct_count} of ${res.total_questions} correct)`);
      if (sessionId) { location.hash = `#/sessions/${enc(sessionId)}`; router(); } else { location.hash = "#/quizzes"; router(); }
    } catch (ex) { btn.disabled = false; $("#qz-status").textContent = ""; err(ex.message); toast("Couldn't save the quiz", "error"); }
  }

  /* ================= Events ================= */
  let searchTimer;
  document.addEventListener("click", async (e) => {
    const el = e.target.closest("[data-action]"); if (!el) return;
    const a = el.dataset.action, f = state.sessionFilters;
    if (a === "open-ingest") openIngest();
    else if (a === "open-quiz") openQuiz(el.dataset.session || "");
    else if (a === "close-dialog") el.closest("dialog").close();
    else if (a === "retry") router();
    else if (a === "add-q") addQuestionRow();
    else if (a === "remove-q") { const rows = $("#qz-rows"); if (rows.children.length > 1) el.closest(".q-row").remove(); }
    else if (a === "copy") { try { await navigator.clipboard.writeText(el.dataset.copy); toast("Copied to clipboard"); } catch { toast("Couldn't copy — select the text and copy it manually.", "error"); } }
    else if (a === "tab") { state.progressTab = el.dataset.tab; router(); }
    else if (a === "theme") { applyTheme(el.dataset.theme); $$("[data-action=theme]").forEach((b) => { const on = b === el; b.setAttribute("aria-checked", on); b.setAttribute("aria-selected", on); }); toast(`Theme: ${el.dataset.theme}`); }
    else if (a === "apply-student") { const v = $("#set-new-student").value; if (!v.trim()) return toast("Enter a student ID first.", "error"); switchStudent(v); }
    else if (a === "page-prev") { f.page = Math.max(0, f.page - 1); loadSessionList(state.token); }
    else if (a === "page-next") { f.page += 1; loadSessionList(state.token); }
    else if (a === "clear-filters") { Object.assign(f, { page: 0, search: "", subject: "", topic: "", cls: "" }); router(); }
    else if (a === "toggle-quiz") {
      const row = $(`#qd-${CSS.escape(el.dataset.id)}`), open = row.hidden;
      row.hidden = !open; el.setAttribute("aria-expanded", open); el.textContent = open ? "Hide answers" : "View answers";
      if (open && !row.dataset.loaded) {
        $("td", row).innerHTML = `<div class="skel" style="height:50px"></div>`;
        try { const q = await getS(`/quizzes/${enc(el.dataset.id)}`); $("td", row).innerHTML = quizItems(q.question_breakdown); row.dataset.loaded = "1"; }
        catch (ex) { $("td", row).innerHTML = `<span style="color:var(--red)">${esc(ex.message)}</span>`; }
      }
    }
  });
  document.addEventListener("input", (e) => {
    if (e.target.dataset.filter !== "search") return;
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { state.sessionFilters.search = e.target.value; state.sessionFilters.page = 0; loadSessionList(state.token); }, 300);
  });
  document.addEventListener("change", (e) => {
    const k = e.target.dataset.filter;
    if (k && k !== "search") { state.sessionFilters[k] = e.target.value; state.sessionFilters.page = 0; loadSessionList(state.token); }
    if (e.target.id === "set-student") switchStudent(e.target.value);
  });
  $("#student-select").addEventListener("change", (e) => switchStudent(e.target.value));
  $("#menu-btn").addEventListener("click", () => setDrawer(true));
  $("#more-btn").addEventListener("click", () => setDrawer(true));
  $("#scrim").addEventListener("click", () => setDrawer(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setDrawer(false); });
  $("#collapse-btn").addEventListener("click", (e) => {
    const c = $("#app").classList.toggle("collapsed");
    localStorage.setItem("edulens_sidebar", c ? "collapsed" : "open");
    e.currentTarget.setAttribute("aria-pressed", c); e.currentTarget.setAttribute("aria-label", c ? "Expand sidebar" : "Collapse sidebar");
  });
  window.addEventListener("hashchange", router);

  /* ================= Init ================= */
  (async function init() {
    if (localStorage.getItem("edulens_sidebar") === "collapsed") { $("#app").classList.add("collapsed"); $("#collapse-btn").setAttribute("aria-pressed", "true"); $("#collapse-btn").setAttribute("aria-label", "Expand sidebar"); }
    renderIdentity();
    try { const r = await api("/students"); state.students = r.students || []; } catch { /* the page still works with the saved ID */ }
    renderIdentity();
    router();
  })();
})();
