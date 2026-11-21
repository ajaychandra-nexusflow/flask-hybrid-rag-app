const $ = (id) => document.getElementById(id);
const history = [];

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

async function health() {
  try {
    const h = await api("/api/health");
    $("health").textContent = `${h.chunks} chunks · ${h.region}`;
    $("health").classList.remove("bad");
  } catch { $("health").textContent = "offline"; $("health").classList.add("bad"); }
}

async function loadDocs() {
  const { documents } = await api("/api/documents");
  $("docs").innerHTML = documents.map((d) =>
    `<li><span>${esc(d.filename)}<br><small>${d.chunks} chunks</small></span><button title="Delete" data-id="${d.doc_id}">×</button></li>`
  ).join("") || '<li class="muted">No documents yet</li>';
}

$("docs").addEventListener("click", async (e) => {
  const id = e.target.dataset && e.target.dataset.id;
  if (!id) return;
  await api(`/api/documents/${id}`, { method: "DELETE" });
  loadDocs(); health();
});

async function upload(files) {
  if (!files.length) return;
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f));
  $("upmsg").textContent = "Indexing... this can take a moment.";
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    $("upmsg").textContent = `Indexed ${r.added.length} file(s)` + (r.skipped.length ? `, skipped ${r.skipped.length}` : "");
  } catch (err) { $("upmsg").textContent = err.message; }
  loadDocs(); health();
}
$("file").addEventListener("change", (e) => { upload(e.target.files); e.target.value = ""; });
const drop = $("drop");
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => upload(e.dataTransfer.files));

function addMsg(cls, html) {
  const log = $("log");
  const empty = log.querySelector(".empty"); if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = `msg ${cls}`; el.innerHTML = html;
  log.appendChild(el); log.scrollTop = log.scrollHeight;
  return el;
}

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("q").value.trim(); if (!q) return;
  $("q").value = ""; $("send").disabled = true;
  addMsg("user", esc(q));
  const pending = addMsg("bot", "Thinking...");
  try {
    const r = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, mode: $("mode").value, top_k: Number($("topk").value), history }),
    });
    history.push({ role: "user", content: q }, { role: "assistant", content: r.answer });
    const src = r.sources.map((s) => `<div class="src"><b>[${s.n}] ${esc(s.filename)}</b> · score ${s.score}<div>${esc(s.snippet)}…</div></div>`).join("");
    pending.innerHTML = esc(r.answer) + (src ? `<div class="sources">${src}</div>` : "");
  } catch (err) {
    pending.classList.add("err"); pending.textContent = err.message;
  }
  $("send").disabled = false; $("log").scrollTop = $("log").scrollHeight;
});

health(); loadDocs();
