// Explorer for the held-out FinanceBench run: data/explorer.json is exported from the eval
// runs by scripts/export_site_data.py. All text is inserted escaped.

const PAGE = 20;
const TYPE_LABEL = {
  "metrics-generated": "metric calculation",
  "domain-relevant": "analyst question",
  "novel-generated": "open question",
};
const OUTCOME_LABEL = { correct: "Correct", wrong: "Wrong", declined: "Declined" };
const FILTERS = {
  all: () => true,
  fixed: (q) => q.baseline.outcome !== "correct" && q.final.outcome === "correct",
  failing: (q) => q.final.outcome !== "correct",
  worse: (q) => q.baseline.outcome === "correct" && q.final.outcome !== "correct",
};

const $ = (sel) => document.querySelector(sel);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

let data = null;
let filter = "all";
let query = "";
let shown = PAGE;

function verdict(outcome, who) {
  return `<span class="v v--${outcome}"><span class="v__who">${who}</span> ${OUTCOME_LABEL[outcome]}</span>`;
}

function sourceList(sources) {
  return `<ol class="src-list">${sources
    .map(
      (s) => `<li>
        <span class="rank">${s.rank}</span>
        <span class="doc">${esc(s.doc)}, ${esc(s.pages)}</span>
        <span class="flags">${s.gold ? '<span class="flag flag--gold">evidence page</span>' : ""}${
          s.cited ? '<span class="flag">cited</span>' : ""
        }</span>
        <details><summary>Show passage</summary><pre>${esc(s.text)}</pre></details>
      </li>`,
    )
    .join("")}</ol>`;
}

function answerCard(title, a) {
  return `<div class="ans ans--${a.outcome}">
    <h4><span>${title}</span>${verdict(a.outcome, "").replace('<span class="v__who"></span> ', "")}</h4>
    <p>${esc(a.text)}</p>
    ${a.judge ? `<p class="judge">Judge&rsquo;s note: ${esc(a.judge)}</p>` : ""}
  </div>`;
}

function questionItem(q) {
  const evidence = q.evidence
    .map((e) => `<span class="flag flag--gold">evidence</span>${esc(e.doc)}, p. ${e.page}`)
    .join("; ");
  return `<li><details class="q">
    <summary>
      <span>
        <span class="q__text">${esc(q.question)}</span>
        <span class="q__meta">${esc(q.filing)}, ${TYPE_LABEL[q.type] ?? esc(q.type)}</span>
      </span>
      <span class="verdicts">${verdict(q.baseline.outcome, "Vector search:")}<span class="arrow" aria-hidden="true">then</span>${verdict(q.final.outcome, "This system:")}</span>
    </summary>
    <div class="q__body">
      <div class="answers">
        <div class="ans ans--gold"><h4><span>Analyst&rsquo;s answer</span></h4><p>${esc(q.gold)}</p></div>
        ${answerCard("This system", q.final)}
        ${answerCard("Plain vector search", q.baseline)}
      </div>
      <p class="evidence">Where the evidence is: ${evidence}</p>
      <div class="sources src-cols">
        <div><h4>Passages this system read</h4>${sourceList(q.final.sources)}</div>
        <div><h4>Passages vector search read</h4>${sourceList(q.baseline.sources)}</div>
      </div>
    </div>
  </details></li>`;
}

function unanswerableItem(u) {
  const outcome = u.declined ? "correct" : "wrong";
  return `<li><details class="q">
    <summary>
      <span>
        <span class="q__text">${esc(u.question)}</span>
        <span class="q__meta">Unanswerable: ${esc(u.category)}</span>
      </span>
      <span class="verdicts"><span class="v v--${outcome}">${u.declined ? "Declined" : "Answered"}</span></span>
    </summary>
    <div class="q__body"><div class="answers">
      <div class="ans ans--${outcome}"><h4><span>This system</span></h4><p>${esc(u.answer)}</p></div>
    </div></div>
  </details></li>`;
}

function matches(text) {
  return !query || text.toLowerCase().includes(query);
}

function render() {
  const list = $("#qlist");
  let items;
  let html;
  if (filter === "unanswerable") {
    items = data.unanswerable.filter((u) => matches(`${u.question} ${u.category}`));
    html = items.slice(0, shown).map(unanswerableItem).join("");
    const declined = items.filter((u) => u.declined).length;
    $("#count").textContent = `${items.length} questions with no answer in the corpus; declined ${declined}.`;
  } else {
    items = data.questions.filter((q) => FILTERS[filter](q) && matches(`${q.question} ${q.filing}`));
    html = items.slice(0, shown).map(questionItem).join("");
    const right = items.filter((q) => q.final.outcome === "correct").length;
    $("#count").textContent = items.length
      ? `${items.length} questions; this system answers ${right} correctly.`
      : "No questions match. Try another company name, or clear the search.";
  }
  list.innerHTML = html;
  const more = document.querySelector(".more");
  if (more) more.remove();
  if (items.length > shown) {
    list.insertAdjacentHTML(
      "afterend",
      `<div class="more"><button type="button">Show ${Math.min(PAGE, items.length - shown)} more</button></div>`,
    );
    document.querySelector(".more button").addEventListener("click", () => {
      shown += PAGE;
      render();
    });
  }
}

function setCounts() {
  for (const btn of document.querySelectorAll(".seg button")) {
    const f = btn.dataset.filter;
    const n = f === "unanswerable" ? data.unanswerable.length : data.questions.filter(FILTERS[f]).length;
    btn.insertAdjacentHTML("beforeend", `<span class="n">${n}</span>`);
  }
}

async function init() {
  try {
    const res = await fetch("data/explorer.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    $("#count").textContent = `The answers couldn't be loaded (${err.message}). Reload the page to try again.`;
    return;
  }
  $("#data-note").textContent = `Source: ${data.source}.`;
  setCounts();
  for (const btn of document.querySelectorAll(".seg button")) {
    btn.addEventListener("click", () => {
      for (const b of document.querySelectorAll(".seg button")) b.setAttribute("aria-checked", String(b === btn));
      filter = btn.dataset.filter;
      shown = PAGE;
      render();
    });
  }
  $("#search").addEventListener("input", (e) => {
    query = e.target.value.trim().toLowerCase();
    shown = PAGE;
    render();
  });
  render();
}

init();
