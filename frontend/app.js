/* CQA — фронтенд без фреймворков. Роутинг по hash: #/, #/c/<id>, #/checklists, #/appeals, #/stats */
(() => {
  "use strict";
  const API = (window.CQA_API_BASE || "") + "/api";
  const $view = document.getElementById("view");
  let routeSeq = 0;
  const stale = (my) => my !== routeSeq;
  const state = { filters: { dataset: "", lang: "", channel: "", q: "" }, job: null, checklist: null, labelMode: false };

  // ---------- helpers ----------
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n) => (n === null || n === undefined ? "—" : Number(n).toLocaleString("ru-RU", { maximumFractionDigits: 2 }));
  async function api(path, opts = {}) {
    const res = await fetch(API + path, {
      headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
      ...opts,
      body: opts.body && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText);
    return data;
  }
  function toast(msg, isError = false) {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.className = "toast" + (isError ? " error" : "");
    clearTimeout(t._h);
    t._h = setTimeout(() => t.classList.add("hidden"), isError ? 6000 : 3000);
  }
  function answerClass(crit, a) {
    if (a === null || a === undefined) return "none";
    if (a === "na") return "na";
    if (crit.type === "flag") return a === "yes" ? "no" : "yes";
    if (crit.type === "scale") {
      const max = Math.max(...crit.scale.map((s) => +s.value));
      const r = +a / max;
      return r >= 0.99 ? "yes" : r >= 0.5 ? "mid" : "no";
    }
    return a === "yes" ? "yes" : "no";
  }
  function answerLabel(crit, a) {
    if (a === null || a === undefined) return "—";
    if (a === "na") return "N/A";
    if (crit.type === "scale") return `${a} из ${Math.max(...crit.scale.map((s) => +s.value))}`;
    if (crit.type === "flag") return a === "yes" ? "Да — нарушение" : "Нет";
    return a === "yes" ? "Да" : "Нет";
  }
  function allowedAnswers(crit) {
    const vals = crit.type === "scale" ? crit.scale.map((s) => String(s.value)) : ["yes", "no"];
    if (crit.na_allowed) vals.push("na");
    return vals;
  }
  function shortAnswer(crit, a) {
    if (a === null || a === undefined) return "·";
    if (a === "na") return "N/A";
    if (crit.type === "scale") return a;
    if (crit.type === "flag") return a === "yes" ? "!" : "✓";
    return a === "yes" ? "Да" : "Нет";
  }
  const scoreColor = (t) => (t === null || t === undefined ? "var(--muted)" : t >= 85 ? "var(--yes)" : t >= 65 ? "var(--mid)" : "var(--no)");
  function modal(html, onMount) {
    const root = document.getElementById("modal-root");
    root.innerHTML = `<div class="modal-back"><div class="modal" role="dialog" aria-modal="true">${html}</div></div>`;
    const back = root.firstElementChild;
    const close = () => (root.innerHTML = "");
    back.addEventListener("click", (e) => e.target === back && close());
    document.addEventListener("keydown", function onKey(e) { if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); } });
    onMount && onMount(back.querySelector(".modal"), close);
  }

  // ---------- header ----------
  async function refreshHeader() {
    try {
      const h = await api("/health");
      document.getElementById("llm-status").innerHTML = h.llm_configured
        ? `Модель: <b>${esc(h.model)}</b>`
        : `<span style="color:#ffd27a">LLM не настроена — заполните .env</span>`;
      const open = await api("/appeals?status=open");
      const b = document.getElementById("appeals-badge");
      b.textContent = open.length;
      b.classList.toggle("hidden", !open.length);
    } catch (e) {
      document.getElementById("llm-status").textContent = "Сервер недоступен";
    }
  }
  async function activeChecklist() {
    const list = await api("/checklists");
    state.checklist = list.find((c) => c.active) || list[0];
    return state.checklist;
  }

  // ---------- list ----------
  async function renderList() {
    const my = routeSeq;
    const [items, datasets, cl] = await Promise.all([api("/conversations"), api("/datasets"), activeChecklist()]);
    if (stale(my)) return;
    const f = state.filters;
    const shown = items.filter((i) =>
      (!f.dataset || i.dataset === f.dataset) && (!f.lang || i.language === f.lang) &&
      (!f.channel || i.channel === f.channel) && (!f.q || i.conversation_id.toLowerCase().includes(f.q.toLowerCase())));
    const evaluated = shown.filter((i) => i.evaluation);
    const avg = evaluated.length ? evaluated.reduce((s, i) => s + (i.evaluation.total || 0), 0) / evaluated.length : null;
    const langs = [...new Set(items.map((i) => i.language).filter(Boolean))];
    const channels = [...new Set(items.map((i) => i.channel).filter(Boolean))];
    const crits = cl.criteria;

    $view.innerHTML = `
      <div class="row mb">
        <div><h1>Диалоги</h1><div class="muted">Активный чек-лист: <b>${esc(cl.name)}</b> · версия ${cl.version} · ${crits.length} критериев</div></div>
        <div class="spacer"></div>
        <div class="card pad row" style="padding:10px 14px">
          <input id="group-id" placeholder="group_id, напр. DEV" style="width:170px">
          <button class="btn" id="fetch-btn">Загрузить из API</button>
          <label class="btn" style="margin:0">Файл JSON<input type="file" id="upload" accept=".json" hidden></label>
        </div>
      </div>
      <div class="card">
        <div class="toolbar">
          <select id="f-dataset"><option value="">Все наборы</option>${datasets.map((d) => `<option ${d === f.dataset ? "selected" : ""}>${esc(d)}</option>`).join("")}</select>
          <select id="f-lang"><option value="">Все языки</option>${langs.map((d) => `<option ${d === f.lang ? "selected" : ""}>${esc(d)}</option>`).join("")}</select>
          <select id="f-channel"><option value="">Все каналы</option>${channels.map((d) => `<option ${d === f.channel ? "selected" : ""}>${esc(d)}</option>`).join("")}</select>
          <input id="f-q" placeholder="Поиск по ID" value="${esc(f.q)}" style="width:150px">
          <span class="muted small">Показано ${shown.length} · оценено ${evaluated.length} · средний балл <b>${fmt(avg)}</b></span>
          <div class="spacer"></div>
          <div id="job-box" class="row"></div>
          <button class="btn" id="eval-missing">Оценить неоценённые</button>
          <button class="btn primary" id="eval-all">Оценить показанные (${shown.length})</button>
          <a class="btn" href="${API}/export${f.dataset ? "?dataset=" + encodeURIComponent(f.dataset) : ""}" target="_blank">Экспорт JSON</a>
        </div>
        ${shown.length ? `<table class="grid"><thead><tr>
          <th>ID</th><th>Набор</th><th>Канал</th><th>Язык</th><th>Итог</th><th>Критерии</th><th></th></tr></thead><tbody>
          ${shown.map((i) => {
            const ev = i.evaluation;
            const dots = crits.map((c) => {
              const a = ev ? ev.answers[c.id] : undefined;
              return `<span class="dot ${answerClass(c, a)}" title="${esc(c.name)}: ${esc(answerLabel(c, a))}">${esc(shortAnswer(c, a))}</span>`;
            }).join("");
            return `<tr data-id="${esc(i.conversation_id)}">
              <td><b>${esc(i.conversation_id)}</b></td>
              <td class="muted">${esc(i.dataset)}</td>
              <td>${i.channel === "voice call" ? "🎧 звонок" : "💬 чат"}</td>
              <td>${esc(i.language || "")}</td>
              <td>${ev ? `<span class="score" style="color:${scoreColor(ev.total)}">${fmt(ev.total)}</span>${ev.overridden ? ` <span class="pill mid" title="Изменено по апелляции">было ${fmt(ev.original_total)}</span>` : ""}` : `<span class="muted">не оценён</span>`}
                  ${ev && ev.gross_error ? ' <span class="pill gross">грубая ошибка</span>' : ""}
                  ${ev && ev.stale ? ' <span class="pill info" title="Оценён другой версией чек-листа">старый чек-лист</span>' : ""}
                  ${ev && ev.errors && ev.errors.length ? ' <span class="pill no">ошибки LLM</span>' : ""}</td>
              <td><div class="dots">${dots}</div></td>
              <td>${i.open_appeals ? `<span class="pill mid">апелляций: ${i.open_appeals}</span>` : ""}</td>
            </tr>`;
          }).join("")}</tbody></table>` : `<div class="empty">Диалогов нет. Загрузите набор по group_id (например, <b>DEV</b>) или файлом.</div>`}
      </div>`;

    const bind = (id, key) => document.getElementById(id).addEventListener(id === "f-q" ? "input" : "change", (e) => {
      state.filters[key] = e.target.value;
      if (id === "f-q") { clearTimeout(bind._t); bind._t = setTimeout(renderList, 250); } else renderList();
    });
    bind("f-dataset", "dataset"); bind("f-lang", "lang"); bind("f-channel", "channel"); bind("f-q", "q");
    $view.querySelectorAll("tbody tr").forEach((tr) => tr.addEventListener("click", () => (location.hash = "#/c/" + encodeURIComponent(tr.dataset.id))));
    document.getElementById("fetch-btn").addEventListener("click", async (e) => {
      const gid = document.getElementById("group-id").value.trim();
      if (!gid) return toast("Введите group_id", true);
      e.target.disabled = true;
      try {
        const r = await api("/datasets/fetch", { method: "POST", body: { group_id: gid } });
        toast(`Загружено ${r.loaded} диалогов в набор ${r.dataset}`);
        state.filters.dataset = r.dataset;
        renderList();
      } catch (err) { toast(err.message, true); e.target.disabled = false; }
    });
    document.getElementById("upload").addEventListener("change", async (e) => {
      const fd = new FormData();
      fd.append("file", e.target.files[0]);
      try {
        const r = await api("/datasets/upload", { method: "POST", body: fd });
        toast(`Загружено ${r.loaded} диалогов в набор ${r.dataset}`);
        state.filters.dataset = r.dataset;
        renderList();
      } catch (err) { toast(err.message, true); }
    });
    const startJob = async (ids, onlyMissing) => {
      try {
        const j = await api("/evaluate", { method: "POST", body: { conversation_ids: ids, only_missing: onlyMissing } });
        if (!j.total) return toast("Всё уже оценено текущей версией чек-листа");
        state.job = j.id;
        pollJob();
      } catch (err) { toast(err.message, true); }
    };
    document.getElementById("eval-all").addEventListener("click", () => startJob(shown.map((i) => i.conversation_id), false));
    document.getElementById("eval-missing").addEventListener("click", () => startJob(shown.map((i) => i.conversation_id), true));
    if (state.job) pollJob();
  }

  async function pollJob() {
    const box = document.getElementById("job-box");
    if (!state.job) return;
    try {
      const j = await api("/jobs/" + state.job);
      if (box) box.innerHTML = `<span class="spinner"></span><span class="small">${j.done}/${j.total} · $${j.cost_usd.toFixed(3)}</span>
        <div class="progress"><div style="width:${j.total ? (j.done / j.total) * 100 : 0}%"></div></div>`;
      if (j.status === "finished") {
        state.job = null;
        toast(`Оценка завершена: ${j.done - j.failed.length} успешно, ошибок ${j.failed.length}, стоимость $${j.cost_usd.toFixed(3)}`, j.failed.length > 0);
        if (location.hash === "" || location.hash === "#/") renderList();
        return;
      }
    } catch (e) { state.job = null; return; }
    clearTimeout(pollJob._t);
    pollJob._t = setTimeout(pollJob, 1500);
  }

  // ---------- conversation card ----------
  async function renderCard(id) {
    const my = routeSeq;
    const [d, gold] = await Promise.all([api("/conversations/" + encodeURIComponent(id)), api("/gold/" + encodeURIComponent(id)).catch(() => null)]);
    const ev = d.evaluation;
    const cl = ev ? ev.checklist_snapshot : await activeChecklist();
    if (stale(my)) return;
    const critDef = Object.fromEntries(cl.criteria.map((c) => [c.id, c]));
    const scoreItem = ev ? Object.fromEntries(ev.final_score.items.map((i) => [i.criterion_id, i])) : {};
    const appealsBy = {};
    d.appeals.forEach((a) => { if (ev && a.evaluation_id === ev.id) (appealsBy[a.criterion_id] ||= []).push(a); });
    const msgIdx = Object.fromEntries(d.messages.map((m, i) => [m.id, i]));

    $view.innerHTML = `
      <div class="row mb">
        <a href="#/" class="btn ghost">← К списку</a>
        <div><h1>${esc(d.conversation_id)}</h1>
          <div class="muted small">${d.channel === "voice call" ? "🎧 Звонок" : "💬 Чат"} · ${esc(d.language || "")} · набор ${esc(d.dataset)} · ${d.messages.length} сообщений</div></div>
        <div class="spacer"></div>
        <label class="row small muted"><input type="checkbox" id="label-mode" ${state.labelMode ? "checked" : ""}> Режим разметки эталона</label>
        <button class="btn primary" id="re-eval">${ev ? "Оценить заново" : "Запустить оценку"}</button>
      </div>
      <div class="layout">
        <div class="card">
          <div class="sticky-head">
            <div class="row"><h2 style="margin:0">Транскрипт</h2><div class="spacer"></div>
              <span class="small muted">Клик по доказательству справа прокрутит к реплике</span></div>
            ${d.has_audio ? `<div style="margin-top:8px"><audio id="player" controls preload="none" src="${API}/conversations/${encodeURIComponent(id)}/audio"></audio>
              <div class="small muted">Запись ведётся с паузами — позиция реплики в записи рассчитана с их учётом. <a href="${esc(d.audio_page)}" target="_blank">Открыть плеер организаторов</a></div></div>` : ""}
          </div>
          <div class="transcript" id="transcript">
            ${d.messages.map((m) => `
              <div class="msg ${esc(m.speaker)}" id="msg-${m.id}" data-id="${m.id}">
                <div class="meta"><div class="mid">${m.id}</div><div title="${esc(m.ts)}">${esc(m.clock)}</div>
                  ${d.has_audio && m.audio_sec !== null && m.audio_sec !== undefined ? `<button class="play" data-sec="${m.audio_sec}" title="Слушать с этого места">▶ ${fmtSec(m.audio_sec)}</button>` : ""}</div>
                <div><div class="who">${{ client: "Клиент", operator: "Оператор", bot: "Бот", system: "Система" }[m.speaker] || esc(m.speaker)}</div>
                  <div class="txt">${esc(m.text)}</div></div>
              </div>`).join("")}
          </div>
        </div>
        <div>
          ${ev ? renderSummary(ev, d) : `<div class="card pad empty">Диалог ещё не оценён. Нажмите «Запустить оценку».</div>`}
          ${ev ? ev.criteria.map((it) => renderCrit(it, critDef[it.criterion_id], scoreItem[it.criterion_id], appealsBy[it.criterion_id] || [], ev, gold)).join("") : ""}
          ${d.history.length > 1 ? `<div class="card pad small muted">История оценок: ${d.history.map((h) => `#${h.id} (${esc(h.checklist_id)} v${h.checklist_version}, ${esc(h.created_at.slice(0, 16).replace("T", " "))})`).join(", ")}</div>` : ""}
        </div>
      </div>`;

    const player = document.getElementById("player");
    const focusMsg = (mid, range) => {
      document.querySelectorAll(".msg.hl, .msg.range").forEach((el) => el.classList.remove("hl", "range"));
      if (range) {
        const [a, b] = [msgIdx[range[0]], msgIdx[range[1]]].sort((x, y) => x - y);
        for (let i = a; i <= b; i++) document.getElementById("msg-" + d.messages[i].id).classList.add("range");
      }
      const el = document.getElementById("msg-" + mid);
      if (!el) return;
      el.classList.add("hl");
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      const m = d.messages[msgIdx[mid]];
      if (player && m && m.audio_sec !== null && m.audio_sec !== undefined) { player.currentTime = Math.max(0, m.audio_sec - 1); }
    };
    $view.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => focusMsg(b.dataset.goto)));
    $view.querySelectorAll("[data-range]").forEach((b) => b.addEventListener("click", () => {
      const [a, z] = b.dataset.range.split(",");
      focusMsg(a, [a, z]);
    }));
    $view.querySelectorAll(".play").forEach((b) => b.addEventListener("click", () => {
      if (!player) return;
      player.currentTime = +b.dataset.sec;
      player.play().catch((e) => toast("Не удалось воспроизвести: " + e.message, true));
    }));
    document.getElementById("label-mode").addEventListener("change", (e) => { state.labelMode = e.target.checked; renderCard(id); });
    document.getElementById("re-eval").addEventListener("click", async (e) => {
      e.target.disabled = true;
      e.target.innerHTML = '<span class="spinner"></span> Оцениваю…';
      try { await api("/evaluate/" + encodeURIComponent(id), { method: "POST" }); toast("Оценка готова"); renderCard(id); }
      catch (err) { toast(err.message, true); e.target.disabled = false; e.target.textContent = "Оценить заново"; }
    });
    $view.querySelectorAll("[data-appeal]").forEach((b) => b.addEventListener("click", () => openAppeal(ev, critDef[b.dataset.appeal], id)));
    $view.querySelectorAll("select[data-gold]").forEach((s) => s.addEventListener("change", async () => {
      const answers = {};
      $view.querySelectorAll("select[data-gold]").forEach((x) => { if (x.value) answers[x.dataset.gold] = x.value; });
      try { await api("/gold/" + encodeURIComponent(id), { method: "PUT", body: { answers } }); toast("Эталон сохранён"); }
      catch (err) { toast(err.message, true); }
    }));
    const hashMsg = new URLSearchParams(location.hash.split("?")[1] || "").get("m");
    if (hashMsg) setTimeout(() => focusMsg(hashMsg), 100);
  }
  const fmtSec = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  function renderSummary(ev, d) {
    const fs = ev.final_score, os = ev.original_score;
    const changed = fs.total !== os.total;
    return `<div class="card pad mb">
      <div class="summary">
        <div style="text-align:center">
          <div class="score big" style="color:${scoreColor(fs.total)}">${fmt(fs.total)}</div>
          <div class="small muted">из ${fs.max_total ?? "—"}</div>
        </div>
        <div>
          <div class="row" style="margin-bottom:8px">
            ${fs.gross_error ? '<span class="pill gross">⚠ Грубая ошибка выявлена</span>' : '<span class="pill yes">Грубых ошибок нет</span>'}
            ${changed ? `<span class="pill mid">Исходная оценка ИИ: ${fmt(os.total)} → итог после апелляций: ${fmt(fs.total)}</span>` : ""}
            ${ev.errors && ev.errors.length ? `<span class="pill no">Не оценены: ${ev.errors.join(", ")}</span>` : ""}
          </div>
          <div class="sections">${fs.sections.map((s) => `<div class="section-box"><div class="small muted">${esc(s.name)}</div>
            <div><b>${fmt(s.points)}</b> / ${fmt(s.max_points)} <span class="small muted">(${fmt(s.percent)}%)</span></div></div>`).join("")}</div>
          <div class="small muted" style="margin-top:8px">Чек-лист ${esc(ev.checklist_id)} v${ev.checklist_version} · модель ${esc(ev.model)} · вызовов LLM ${ev.llm_calls} ·
            ${ev.usage.input + ev.usage.output} токенов · ≈ $${(ev.cost_usd || 0).toFixed(4)} · ${ev.duration_s} с
            ${state.checklist && (state.checklist.id !== ev.checklist_id || state.checklist.version !== ev.checklist_version) ? ' · <b style="color:var(--mid)">оценено не текущей версией чек-листа</b>' : ""}</div>
        </div>
      </div>
    </div>`;
  }

  function renderCrit(it, c, si, appeals, ev, gold) {
    const final = ev.final_answers[it.criterion_id];
    const original = ev.original_answers[it.criterion_id];
    const overridden = final !== original;
    const open = appeals.find((a) => a.status === "open");
    const pts = si && si.applicable
      ? `<b>${fmt(si.points)}</b> / ${fmt(si.effective_weight)} балл. <span class="small">(вес ${fmt(c.weight)})</span>`
      : c.type === "flag" ? "без баллов" : `исключён, вес ${fmt(c.weight)} перераспределён`;
    const goldVal = gold && gold.answers ? gold.answers[c.id] : "";
    return `<div class="crit ${overridden ? "overridden" : ""}">
      <div class="crit-head">
        <span class="crit-name">${esc(c.name)}</span>
        <span class="pill ${answerClass(c, final)}">${esc(answerLabel(c, final))}</span>
        ${overridden ? `<span class="pill na" title="Исходная оценка ИИ">ИИ: ${esc(answerLabel(c, original))}</span>` : ""}
        ${it.confidence === "low" ? '<span class="pill mid" title="Модель не уверена — стоит проверить">низкая уверенность</span>' : ""}
        ${it.votes ? `<span class="pill info" title="Голоса прогонов">${esc(Object.entries(it.votes).map(([k, v]) => answerLabel(c, k) + "×" + v).join(", "))}</span>` : ""}
        <span class="crit-points">${pts}</span>
      </div>
      ${it.error ? `<div class="appeal-note">Ошибка LLM: ${esc(it.error)}</div>` : ""}
      <div class="rationale">${esc(it.rationale)}</div>
      ${final === "na" ? `<div class="small muted">Условие N/A: ${esc(c.na_condition)}</div>` : ""}
      <div class="evidence">
        ${(it.evidence || []).map((e) => `<button class="ev" data-goto="${esc(e.msg_id)}"><span class="mid">${esc(e.msg_id)}</span>
          <span>«${esc(e.quote)}»</span>${e.why ? `<span class="why">— ${esc(e.why)}</span>` : ""}</button>`).join("")}
        ${it.checked_range ? `<button class="ev" data-range="${esc(it.checked_range.from_msg)},${esc(it.checked_range.to_msg)}">
          <span class="mid">${esc(it.checked_range.from_msg)}–${esc(it.checked_range.to_msg)}</span><span class="why">проверенный участок разговора</span></button>` : ""}
      </div>
      ${it.analysis ? `<details class="analysis"><summary>Ход рассуждения модели</summary><p>${esc(it.analysis)}</p></details>` : ""}
      ${appeals.map((a) => `<div class="appeal-note"><b>Апелляция #${a.id}</b> · ${a.status === "open" ? "ждёт решения" : a.status === "changed" ? "оценка изменена" : "оценка подтверждена"}<br>
        Комментарий: ${esc(a.data.comment)}
        ${a.status !== "open" ? `<br>Решение: ${esc(answerLabel(c, a.data.disputed_answer))} → <b>${esc(answerLabel(c, a.data.new_answer))}</b>. Причина: ${esc(a.data.reason)}<br>
        Итог: ${fmt(a.data.total_before)} → <b>${fmt(a.data.total_after)}</b>` : ""}</div>`).join("")}
      <div class="row" style="margin-top:8px">
        ${!open ? `<button class="btn sm" data-appeal="${esc(c.id)}">Оспорить</button>` : `<a class="btn sm" href="#/appeals">Открыть в очереди проверяющего</a>`}
        ${state.labelMode ? `<span class="spacer"></span><label class="small muted">Эталон:
          <select data-gold="${esc(c.id)}"><option value="">—</option>${allowedAnswers(c).map((v) => `<option value="${v}" ${String(goldVal) === v ? "selected" : ""}>${esc(answerLabel(c, v))}</option>`).join("")}</select></label>` : ""}
      </div>
    </div>`;
  }

  function openAppeal(ev, c, convId) {
    modal(`<h2>Апелляция: ${esc(c.name)}</h2>
      <div class="kv mb"><span class="muted">Текущая оценка</span><b>${esc(answerLabel(c, ev.final_answers[c.id]))}</b>
        <span class="muted">Итог диалога</span><b>${fmt(ev.final_score.total)}</b></div>
      <label class="field mb">Какую оценку вы считаете верной (необязательно)
        <select id="ap-req"><option value="">—</option>${allowedAnswers(c).map((v) => `<option value="${v}">${esc(answerLabel(c, v))}</option>`).join("")}</select></label>
      <label class="field">Комментарий — почему оценка неверна
        <textarea id="ap-comment" rows="4" placeholder="Например: оператор представился в m9 — «Меня зовут Алина»"></textarea></label>
      <div class="actions"><button class="btn" data-close>Отмена</button><button class="btn primary" id="ap-send">Подать апелляцию</button></div>`,
    (m, close) => {
      m.querySelector("[data-close]").addEventListener("click", close);
      m.querySelector("#ap-comment").focus();
      m.querySelector("#ap-send").addEventListener("click", async () => {
        try {
          await api("/appeals", { method: "POST", body: { evaluation_id: ev.id, criterion_id: c.id,
            comment: m.querySelector("#ap-comment").value, requested_answer: m.querySelector("#ap-req").value || null } });
          close(); toast("Апелляция сохранена и передана проверяющему"); refreshHeader(); renderCard(convId);
        } catch (err) { toast(err.message, true); }
      });
    });
  }

  // ---------- appeals (reviewer) ----------
  async function renderAppeals() {
    const my = routeSeq;
    const all = await api("/appeals");
    const open = all.filter((a) => a.status === "open"), done = all.filter((a) => a.status !== "open");
    const convCache = {};
    for (const a of open) if (!convCache[a.conversation_id]) convCache[a.conversation_id] = await api("/conversations/" + encodeURIComponent(a.conversation_id));
    if (stale(my)) return;
    const fragment = (a) => {
      const d = convCache[a.conversation_id];
      const ids = new Set((a.criterion?.evidence || []).map((e) => e.msg_id));
      const r = a.criterion?.checked_range;
      const idx = d.messages.map((m, i) => (ids.has(m.id) ? i : -1)).filter((i) => i >= 0);
      if (r) { idx.push(d.messages.findIndex((m) => m.id === r.from_msg), d.messages.findIndex((m) => m.id === r.to_msg)); }
      if (!idx.length) return "";
      const lo = Math.max(0, Math.min(...idx) - 1), hi = Math.min(d.messages.length - 1, Math.max(...idx) + 1);
      return d.messages.slice(lo, hi + 1).slice(0, 14).map((m) => `<div class="msg ${esc(m.speaker)} ${ids.has(m.id) ? "hl" : ""}">
        <div class="meta"><div class="mid">${m.id}</div><div>${esc(m.clock)}</div></div>
        <div><div class="who">${esc(m.speaker)}</div><div class="txt">${esc(m.text)}</div></div></div>`).join("");
    };
    $view.innerHTML = `<h1>Апелляции</h1><div class="muted mb">Рабочее место проверяющего: спорная оценка, комментарий и фрагмент диалога.</div>
      <h2>Ожидают решения (${open.length})</h2>
      ${open.length ? open.map((a) => `<div class="card pad mb">
        <div class="row"><b>#${a.id} · <a href="#/c/${encodeURIComponent(a.conversation_id)}">${esc(a.conversation_id)}</a> · ${esc(a.data.criterion_name)}</b>
          <span class="pill ${answerClass(a.criterion_def, a.data.disputed_answer)}">ИИ: ${esc(answerLabel(a.criterion_def, a.data.disputed_answer))}</span>
          ${a.data.requested_answer ? `<span class="pill info">Оператор просит: ${esc(answerLabel(a.criterion_def, a.data.requested_answer))}</span>` : ""}
          <span class="spacer"></span><span class="small muted">Итог сейчас: <b>${fmt(a.data.total_before)}</b></span></div>
        <div class="quote"><b>Комментарий:</b> ${esc(a.data.comment)}</div>
        <div class="small"><b>Обоснование ИИ:</b> ${esc(a.criterion?.rationale || "")}</div>
        <div class="small muted" style="margin:4px 0">Правило: ${esc(a.criterion_def.rule)}</div>
        <div style="max-height:320px;overflow:auto;margin:8px 0">${fragment(a)}</div>
        <div class="row">
          <select data-new="${a.id}">${a.allowed_answers.map((v) => `<option value="${v}" ${v === a.data.disputed_answer ? "selected" : ""}>${esc(answerLabel(a.criterion_def, v))}</option>`).join("")}</select>
          <input data-reason="${a.id}" placeholder="Причина решения (обязательно)" style="flex:1;min-width:240px">
          <button class="btn" data-confirm="${a.id}">Подтвердить оценку ИИ</button>
          <button class="btn primary" data-change="${a.id}">Изменить оценку</button>
        </div></div>`).join("") : `<div class="card empty mb">Открытых апелляций нет</div>`}
      <h2>Рассмотренные (${done.length})</h2>
      ${done.length ? `<div class="card"><table class="grid"><thead><tr><th>#</th><th>Диалог</th><th>Критерий</th><th>Решение</th><th>Причина</th><th>Итог</th></tr></thead><tbody>
        ${done.map((a) => `<tr onclick="location.hash='#/c/${encodeURIComponent(a.conversation_id)}'"><td>${a.id}</td><td>${esc(a.conversation_id)}</td><td>${esc(a.data.criterion_name)}</td>
          <td>${a.status === "changed" ? `${esc(answerLabel(a.criterion_def, a.data.disputed_answer))} → <b>${esc(answerLabel(a.criterion_def, a.data.new_answer))}</b>` : "подтверждено"}</td>
          <td>${esc(a.data.reason)}</td><td>${fmt(a.data.total_before)} → <b>${fmt(a.data.total_after)}</b></td></tr>`).join("")}</tbody></table></div>` : `<div class="card empty">Пока пусто</div>`}`;

    const decide = async (id, decision) => {
      const reason = $view.querySelector(`[data-reason="${id}"]`).value;
      const new_answer = $view.querySelector(`[data-new="${id}"]`).value;
      try {
        const r = await api(`/appeals/${id}/decision`, { method: "POST", body: { decision, new_answer, reason } });
        toast(`Решение сохранено. Итог: ${fmt(r.appeal.data.total_before)} → ${fmt(r.appeal.data.total_after)}`);
        refreshHeader(); renderAppeals();
      } catch (err) { toast(err.message, true); }
    };
    $view.querySelectorAll("[data-confirm]").forEach((b) => b.addEventListener("click", () => decide(b.dataset.confirm, "confirm")));
    $view.querySelectorAll("[data-change]").forEach((b) => b.addEventListener("click", () => decide(b.dataset.change, "change")));
  }

  // ---------- checklist editor ----------
  let draft = null;
  async function renderChecklists() {
    const my = routeSeq;
    const list = await api("/checklists");
    if (stale(my)) return;
    const active = list.find((c) => c.active) || list[0];
    if (!draft || draft._src !== active.id + ":" + active.version) {
      draft = JSON.parse(JSON.stringify(active));
      draft._src = active.id + ":" + active.version;
    }
    drawEditor(list, active);
  }

  function drawEditor(list, active) {
    const sum = draft.criteria.filter((c) => c.type !== "flag").reduce((s, c) => s + (+c.weight || 0), 0);
    $view.innerHTML = `
      <div class="row mb">
        <div><h1>Чек-лист</h1><div class="muted">Сохранение создаёт новую версию. Прошлые оценки остаются привязаны к своей версии.</div></div>
        <div class="spacer"></div>
        <select id="cl-switch">${list.map((c) => `<option value="${esc(c.id)}" ${c.id === active.id ? "selected" : ""}>${esc(c.name)} (v${c.version})</option>`).join("")}</select>
        <label class="btn" style="margin:0">Импорт JSON<input type="file" id="cl-import" accept=".json" hidden></label>
        <button class="btn" id="cl-export">Экспорт JSON</button>
      </div>
      <div class="card pad mb">
        <div class="row">
          <label class="field" style="flex:1">Название<input id="cl-name" value="${esc(draft.name)}"></label>
          <label class="field">ID<input id="cl-id" value="${esc(draft.id)}" style="width:220px"></label>
          <label class="field">Округление итога<input id="cl-round" type="number" min="0" max="4" value="${draft.rounding ?? 2}" style="width:110px"></label>
        </div>
        <div class="row" style="margin-top:10px">
          <span class="weights">Сумма весов: <b>${fmt(sum)}</b> ${sum !== 100 ? '<span class="pill mid">не 100 — итог всё равно нормируется к 100</span>' : '<span class="pill yes">100</span>'}</span>
          <span class="muted small">Секции: ${draft.sections.map((s) => esc(s.name)).join(" · ")}</span>
          <button class="btn sm" id="sec-edit">Изменить секции</button>
          <div class="spacer"></div>
          <span class="muted small">Версии: ${(active.versions || []).map((v) => "v" + v.version).join(", ")}</span>
        </div>
      </div>
      <div id="crit-list">${draft.criteria.map((c, i) => critEditor(c, i)).join("")}</div>
      <div class="row mb"><button class="btn" id="add-crit">+ Добавить критерий</button></div>
      <div class="card pad row" style="position:sticky;bottom:10px">
        <span class="muted small">Изменения применяются к новым оценкам после сохранения.</span><div class="spacer"></div>
        <button class="btn" id="cl-reset">Отменить изменения</button>
        <button class="btn" id="cl-save">Сохранить новую версию</button>
        <button class="btn primary" id="cl-save-apply">Сохранить и переоценить диалоги</button>
      </div>`;

    const rerender = () => drawEditor(list, active);
    $view.querySelectorAll("[data-f]").forEach((el) => el.addEventListener(el.type === "checkbox" || el.tagName === "SELECT" ? "change" : "input", () => {
      const c = draft.criteria[+el.dataset.i];
      const f = el.dataset.f;
      c[f] = el.type === "checkbox" ? el.checked : el.type === "number" ? +el.value : el.value;
      if (f === "type") {
        if (c.type === "scale" && !c.scale) c.scale = [{ value: 10, label: "Полностью" }, { value: 5, label: "Частично" }, { value: 0, label: "Нет" }];
        if (c.type === "flag") { c.weight = 0; c.na_allowed = false; }
        rerender();
      }
      if (f === "weight" || f === "na_allowed") rerender();
    }));
    $view.querySelectorAll("[data-sf]").forEach((el) => el.addEventListener("input", () => {
      const s = draft.criteria[+el.dataset.i].scale[+el.dataset.j];
      s[el.dataset.sf] = el.dataset.sf === "value" ? +el.value : el.value;
    }));
    $view.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", () => {
      const i = +b.dataset.i, arr = draft.criteria;
      if (b.dataset.act === "del" && confirm(`Удалить критерий «${arr[i].name}»?`)) arr.splice(i, 1);
      if (b.dataset.act === "up" && i > 0) [arr[i - 1], arr[i]] = [arr[i], arr[i - 1]];
      if (b.dataset.act === "down" && i < arr.length - 1) [arr[i + 1], arr[i]] = [arr[i], arr[i + 1]];
      if (b.dataset.act === "add-step") arr[i].scale.push({ value: 0, label: "" });
      if (b.dataset.act === "del-step") arr[i].scale.splice(+b.dataset.j, 1);
      rerender();
    }));
    document.getElementById("add-crit").addEventListener("click", () => {
      let n = draft.criteria.length + 1, id;
      do { id = "c" + String(n++).padStart(2, "0"); } while (draft.criteria.some((c) => c.id === id));
      draft.criteria.push({ id, section: draft.sections[0]?.id || "", name: "Новый критерий", type: "binary", weight: 10, na_allowed: false, rule: "Да — ", na_condition: "", guidance: "" });
      rerender();
      window.scrollTo(0, document.body.scrollHeight);
    });
    document.getElementById("cl-name").addEventListener("input", (e) => (draft.name = e.target.value));
    document.getElementById("cl-id").addEventListener("input", (e) => (draft.id = e.target.value.trim()));
    document.getElementById("cl-round").addEventListener("input", (e) => (draft.rounding = +e.target.value));
    document.getElementById("cl-reset").addEventListener("click", () => { draft = null; renderChecklists(); });
    document.getElementById("cl-switch").addEventListener("change", async (e) => { await api(`/checklists/${encodeURIComponent(e.target.value)}/activate`, { method: "POST" }); draft = null; renderChecklists(); toast("Активный чек-лист переключён"); });
    document.getElementById("cl-export").addEventListener("click", () => {
      const { _src, active: _a, versions, ...clean } = draft;
      const blob = new Blob([JSON.stringify(clean, null, 2)], { type: "application/json" });
      const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${draft.id}.json`; a.click();
    });
    document.getElementById("cl-import").addEventListener("change", async (e) => {
      try { const cl = JSON.parse(await e.target.files[0].text()); draft = { ...cl, _src: draft._src }; rerender(); toast("Импортировано — проверьте и сохраните"); }
      catch (err) { toast("Не удалось прочитать файл: " + err.message, true); }
    });
    document.getElementById("sec-edit").addEventListener("click", () => {
      modal(`<h2>Секции</h2><p class="muted small">Формат: id | название, по одной на строку.</p>
        <textarea id="sec-text" rows="6">${esc(draft.sections.map((s) => `${s.id} | ${s.name}`).join("\n"))}</textarea>
        <div class="actions"><button class="btn" data-close>Отмена</button><button class="btn primary" id="sec-ok">Готово</button></div>`, (m, close) => {
        m.querySelector("[data-close]").addEventListener("click", close);
        m.querySelector("#sec-ok").addEventListener("click", () => {
          draft.sections = m.querySelector("#sec-text").value.split("\n").map((l) => l.split("|").map((x) => x.trim())).filter((p) => p[0]).map(([id, name]) => ({ id, name: name || id }));
          close(); rerender();
        });
      });
    });
    const save = async (apply) => {
      const { _src, active: _a, versions, ...body } = draft;
      try {
        const saved = await api("/checklists", { method: "POST", body });
        draft = null;
        toast(`Сохранено: ${saved.name} v${saved.version}`);
        if (apply) {
          const ds = state.filters.dataset;
          const j = await api("/evaluate", { method: "POST", body: { dataset: ds || null, only_missing: true } });
          state.job = j.id;
          location.hash = "#/";
          toast(`Чек-лист v${saved.version} применяется к ${j.total} диалогам${ds ? " набора " + ds : ""}`);
        } else renderChecklists();
      } catch (err) { toast(err.message, true); }
    };
    document.getElementById("cl-save").addEventListener("click", () => save(false));
    document.getElementById("cl-save-apply").addEventListener("click", () => save(true));
  }

  function critEditor(c, i) {
    return `<div class="crit-edit">
      <div class="grid2">
        <label class="field">ID<input data-i="${i}" data-f="id" value="${esc(c.id)}"></label>
        <label class="field">Название<input data-i="${i}" data-f="name" value="${esc(c.name)}"></label>
        <label class="field">Секция<select data-i="${i}" data-f="section">${draft.sections.map((s) => `<option value="${esc(s.id)}" ${s.id === c.section ? "selected" : ""}>${esc(s.name)}</option>`).join("")}</select></label>
        <label class="field">Тип ответа<select data-i="${i}" data-f="type">
          <option value="binary" ${c.type === "binary" ? "selected" : ""}>Да / Нет</option>
          <option value="scale" ${c.type === "scale" ? "selected" : ""}>Шкала</option>
          <option value="flag" ${c.type === "flag" ? "selected" : ""}>Флаг (без баллов)</option></select></label>
        <label class="field">Вес<input type="number" min="0" data-i="${i}" data-f="weight" value="${c.weight}" ${c.type === "flag" ? "disabled" : ""}></label>
        <div class="row" style="gap:4px">
          <button class="btn sm ghost" data-act="up" data-i="${i}" title="Выше">↑</button>
          <button class="btn sm ghost" data-act="down" data-i="${i}" title="Ниже">↓</button>
          <button class="btn sm ghost danger" data-act="del" data-i="${i}" title="Удалить">✕</button>
        </div>
      </div>
      <div class="grid1">
        <label class="field">Правило (что значит «Да» / как ставить балл)<textarea data-i="${i}" data-f="rule" rows="2">${esc(c.rule)}</textarea></label>
        <label class="field">Уточнения для ИИ (необязательно)<textarea data-i="${i}" data-f="guidance" rows="2">${esc(c.guidance || "")}</textarea></label>
      </div>
      <div class="row" style="margin-top:10px">
        ${c.type !== "flag" ? `<label class="row small"><input type="checkbox" data-i="${i}" data-f="na_allowed" ${c.na_allowed ? "checked" : ""}> Разрешён N/A</label>` : ""}
        ${c.na_allowed ? `<input data-i="${i}" data-f="na_condition" value="${esc(c.na_condition || "")}" placeholder="Когда ставить N/A" style="flex:1">` : ""}
      </div>
      ${c.type === "scale" ? `<div style="margin-top:10px"><div class="small muted">Ступени шкалы (верхняя ступень = полный вес):</div>
        ${(c.scale || []).map((s, j) => `<div class="scale-row"><input type="number" data-i="${i}" data-j="${j}" data-sf="value" value="${s.value}">
          <input data-i="${i}" data-j="${j}" data-sf="label" value="${esc(s.label)}" placeholder="Описание ступени">
          <button class="btn sm ghost danger" data-act="del-step" data-i="${i}" data-j="${j}">✕</button></div>`).join("")}
        <button class="btn sm" data-act="add-step" data-i="${i}" style="margin-top:6px">+ ступень</button></div>` : ""}
    </div>`;
  }

  // ---------- stats / accuracy ----------
  async function renderStats() {
    const my = routeSeq;
    const [items, cl, report] = await Promise.all([api("/conversations"), activeChecklist(), api("/gold/report").catch(() => null)]);
    if (stale(my)) return;
    const ev = items.filter((i) => i.evaluation);
    const byLang = {};
    ev.forEach((i) => { (byLang[i.language] ||= []).push(i.evaluation.total); });
    const avg = (a) => (a.length ? a.reduce((s, x) => s + (x || 0), 0) / a.length : null);
    $view.innerHTML = `<h1>Сводка</h1><div class="muted mb">Распределение оценок и точность относительно эталонной разметки команды.</div>
      <div class="card pad mb"><h2>Средний балл</h2><div class="row">
        <div class="section-box"><div class="small muted">Все (${ev.length})</div><b>${fmt(avg(ev.map((i) => i.evaluation.total)))}</b></div>
        ${Object.entries(byLang).map(([l, a]) => `<div class="section-box"><div class="small muted">${esc(l)} (${a.length})</div><b>${fmt(avg(a))}</b></div>`).join("")}
        <div class="section-box"><div class="small muted">С грубой ошибкой</div><b>${ev.filter((i) => i.evaluation.gross_error).length}</b></div>
      </div></div>
      <div class="card mb"><div class="pad"><h2>Ответы по критериям</h2></div><table class="grid"><thead><tr><th>Критерий</th><th>Распределение ответов</th></tr></thead><tbody>
        ${cl.criteria.map((c) => {
          const cnt = {};
          ev.forEach((i) => { const a = i.evaluation.answers[c.id]; if (a !== undefined) cnt[a] = (cnt[a] || 0) + 1; });
          return `<tr><td>${esc(c.name)}</td><td>${Object.entries(cnt).map(([a, n]) => `<span class="pill ${answerClass(c, a)}">${esc(answerLabel(c, a))}: ${n}</span>`).join(" ")}</td></tr>`;
        }).join("")}</tbody></table></div>
      ${report && report.n ? `<div class="card mb"><div class="pad"><h2>Точность против эталона (${report.n} диалогов)</h2>
        <div class="row"><div class="section-box"><div class="small muted">Совпадение ответов</div><b>${fmt(report.answer_accuracy * 100)}%</b></div>
        <div class="section-box"><div class="small muted">Средняя ошибка итога (MAE)</div><b>${fmt(report.total_mae)}</b></div></div></div>
        <table class="grid"><thead><tr><th>Критерий</th><th>Совпадение</th><th>Расхождения</th></tr></thead><tbody>
        ${report.per_criterion.map((r) => `<tr><td>${esc(r.name)}</td><td>${fmt(r.accuracy * 100)}% (${r.match}/${r.n})</td>
          <td class="small">${r.mismatches.map((m) => `<a href="#/c/${encodeURIComponent(m.conversation_id)}">${esc(m.conversation_id)}</a>: ИИ ${esc(m.pred)} / эталон ${esc(m.gold)}`).join("<br>")}</td></tr>`).join("")}
        </tbody></table></div>` : `<div class="card pad muted">Эталонной разметки пока нет. Откройте диалог, включите «Режим разметки эталона» и выставьте правильные ответы — здесь появится точность.</div>`}`;
  }

  // ---------- router ----------
  async function route() {
    routeSeq++;
    const h = location.hash || "#/";
    const tab = h.startsWith("#/c/") ? "list" : h.startsWith("#/checklists") ? "checklists" : h.startsWith("#/appeals") ? "appeals" : h.startsWith("#/stats") ? "stats" : "list";
    document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.tab === tab));
    try {
      if (h.startsWith("#/c/")) await renderCard(decodeURIComponent(h.slice(4).split("?")[0]));
      else if (tab === "checklists") await renderChecklists();
      else if (tab === "appeals") await renderAppeals();
      else if (tab === "stats") await renderStats();
      else await renderList();
    } catch (e) {
      $view.innerHTML = `<div class="card pad empty">Ошибка: ${esc(e.message)}</div>`;
    }
  }
  window.addEventListener("hashchange", () => { window.scrollTo(0, 0); route(); });
  refreshHeader();
  activeChecklist().catch(() => {});
  route();
})();
