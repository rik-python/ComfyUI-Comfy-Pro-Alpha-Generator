/**
 * Comfy Pro Alpha Generator: picker + result viewer + About panel.
 *
 * Picker: scrub the plate, type what to find ("person", "dog, bike"...), Find (SAM3 text detection on this
 * frame), click a subject = target (other detections near it are excluded automatically),
 * Alt-click = cut a part (someone else's hand),
 * Shift-click = add a part (own hand SAM3 missed), Ctrl-click = remove the nearest cut/add.
 * Viewer: after a run, wipe between plate and matte comp, scrub, QC badge.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "ComfyProAlphaGenerator";
const LEGACY = "SelectiveMatte";  // old node id: workflows saved before the rename
const TITLE = "Comfy Pro Alpha Generator";
const VERSION = "0.5.4";
const ADVANCED = ["text_encoder", "chunk", "overlap", "fast_attention",
    "seed", "control_after_generate", "erode", "dilate", "near_px", "stabilize_radius", "sam3_threshold"];
const HIDDEN = ["picks", "render"];
// Overlap (v13) tuning: listed with the other Advanced settings, greyed out while the v13 switch is off
const V13_ADVANCED = ["attn_grow", "attn_blur", "attn_strength", "occluder_reach", "limb_cap", "front_smoothing"];
// Comfy Compositing brand (comfycompositing.com), adapted to ComfyUI's dark canvas.
const BRAND = {
    navy: "#12122a", navy2: "#1a1a36", navy3: "#24244a", line: "#34345e", text: "#ececf6", muted: "#9696b8",
    blue: "#63b3ed", purple: "#9a75ea", amber: "#f6a94a", good: "#5dd39e", bad: "#f26d7d",
};
const COLORS = [BRAND.blue, BRAND.purple, "#5fd4c8", "#e6d94a", "#ff7eb6", "#8bd450", "#c3a6ff", "#9aa4b2"];
const TARGET = BRAND.amber, CUT = BRAND.bad, ADD = BRAND.good;
const MAX_OTHERS = 8;  // excluded detections (one SAM3 track each)

const SANS = `"Hanken Grotesk", "Segoe UI", system-ui, sans-serif`;
const MONO = `"JetBrains Mono", "Cascadia Mono", Consolas, monospace`;
const css = `
.sm-root{display:flex;flex-direction:column;gap:7px;font:12px/1.35 ${SANS};color:${BRAND.text};width:100%;height:100%;box-sizing:border-box}
.sm-bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.sm-root [hidden]{display:none!important}
.sm-sp{flex:1 1 0;min-width:0}
.sm-stepper{display:flex;align-items:center;gap:8px;min-height:30px}
.sm-st{display:flex;align-items:center;gap:7px;font:600 12.5px/1 ${SANS};color:${BRAND.muted};white-space:nowrap}
.sm-st-dot{width:20px;height:20px;border-radius:50%;display:grid;place-items:center;box-sizing:border-box;border:1.5px solid ${BRAND.line};font:700 10.5px/1 ${SANS};color:${BRAND.muted}}
.sm-st.now{color:${BRAND.text}}.sm-st.now .sm-st-dot{background:${BRAND.purple};border-color:${BRAND.purple};color:${BRAND.navy}}
.sm-st.done .sm-st-dot{border-color:${BRAND.good};color:${BRAND.good};font-size:0}.sm-st.done .sm-st-dot::before{content:"✓";font-size:12px}
.sm-st-line{flex:0 1 40px;min-width:12px;height:0;border-top:1.5px solid ${BRAND.line}}
.sm-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.sm-ov-tl,.sm-ov-tr{position:absolute;top:10px;z-index:2}.sm-ov-tl{left:10px}.sm-ov-tr{right:10px}
.sm-empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;padding:20px;text-align:center}
.sm-empty-title{font:700 16px/1.2 ${SANS};color:${BRAND.text}}
.sm-empty-text{font:12.5px/1.45 ${SANS};color:${BRAND.muted};max-width:46ch}
.sm-btn.big{--h:38px;padding:0 20px;font-size:13.5px}
.sm-btn.icon{padding:0;width:30px}
.sm-btn,.sm-find,.sm-seg{--h:30px;height:var(--h);box-sizing:border-box;border-radius:8px}
.sm-btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;padding:0 12px;background:${BRAND.navy2};border:1px solid ${BRAND.line};color:${BRAND.text};cursor:pointer;font:600 12px/1 ${SANS};white-space:nowrap;text-decoration:none}
.sm-btn svg{width:14px;height:14px;flex:0 0 14px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;opacity:.85}
.sm-btn:hover{background:${BRAND.navy3};border-color:#4a4a80}
.sm-btn.primary{background:${BRAND.amber};border-color:${BRAND.amber};color:${BRAND.navy}}
.sm-btn.primary svg{fill:currentColor;stroke:none;opacity:1}
.sm-btn.primary:hover{filter:brightness(1.07)}
.sm-btn.quiet{background:transparent;border-color:transparent;color:${BRAND.muted}}
.sm-btn.quiet:hover{background:${BRAND.navy2};color:${BRAND.text}}
.sm-btn:disabled{opacity:.4;cursor:default;filter:none}
.sm-btn.primary:disabled{background:${BRAND.navy2};border-color:${BRAND.line};color:${BRAND.text}}
.sm-btn.primary:disabled svg{fill:currentColor}
.sm-btn:disabled:hover{background:${BRAND.navy2};border-color:${BRAND.line}}
.sm-btn:focus-visible,.sm-seg button:focus-visible,.sm-about summary:focus-visible{outline:2px solid ${BRAND.purple};outline-offset:2px}
.sm-find{width:240px;max-width:calc(100% - 20px);display:flex;align-items:stretch;border:1px solid ${BRAND.line};background:${BRAND.navy};overflow:hidden}
.sm-find:focus-within{border-color:${BRAND.purple}}
.sm-input{flex:1;min-width:0;background:transparent;border:0;color:${BRAND.text};padding:0 10px;font:12.5px/1 ${SANS};outline:none}
.sm-input::placeholder{color:${BRAND.muted}}
.sm-find .sm-btn{height:auto;border:0;border-left:1px solid ${BRAND.line};border-radius:0}
.sm-view{position:relative;flex:1 1 auto;min-height:120px;background:#0b0b1d;border:1px solid ${BRAND.line};border-radius:10px;overflow:hidden}
.sm-view canvas{position:absolute;inset:0;width:100%;height:100%}
.sm-hint{position:absolute;left:8px;bottom:8px;right:8px;background:#0b0b1dcc;border-radius:8px;padding:5px 9px;pointer-events:none}
.sm-chip{display:inline-flex;gap:5px;align-items:center;background:${BRAND.navy};border:1px solid ${BRAND.line};border-radius:999px;padding:1px 9px;font:10.5px/1.6 ${MONO};color:${BRAND.muted}}
.sm-dot{width:7px;height:7px;border-radius:50%}
.sm-scrub{display:flex;gap:8px;align-items:center;font:10.5px ${MONO};color:${BRAND.muted}}.sm-scrub input{flex:1;accent-color:${BRAND.purple}}
.sm-seg{display:inline-flex;gap:2px;background:${BRAND.navy};border:1px solid ${BRAND.line};padding:2px}
.sm-seg button{border:0;background:none;padding:0 14px;border-radius:6px;cursor:pointer;font:600 12px/1 ${SANS};color:${BRAND.muted}}
.sm-seg button:hover{color:${BRAND.text}}
.sm-seg button.on{background:${BRAND.purple};color:${BRAND.navy}}
.sm-qc{display:flex;gap:6px;flex-wrap:wrap}
.sm-badge{--c:${BRAND.muted};border:1px solid var(--c);color:var(--c);background:color-mix(in srgb,var(--c) 12%,transparent);border-radius:999px;padding:1px 9px;font:600 10.5px/1.6 ${MONO}}
.sm-g{--c:${BRAND.good}}.sm-a{--c:${BRAND.amber}}.sm-r{--c:${BRAND.bad}}
.sm-about{flex:0 0 auto;position:relative;background:linear-gradient(180deg,${BRAND.navy3},${BRAND.navy2});border:1px solid ${BRAND.line};border-radius:10px;color:${BRAND.muted};overflow:hidden}
.sm-about::before{content:"";position:absolute;left:0;right:0;top:0;height:3px;background:linear-gradient(90deg,${BRAND.blue},${BRAND.purple},${BRAND.amber})}
.sm-about summary{cursor:pointer;list-style:none;user-select:none;display:flex;align-items:center;gap:12px;padding:13px 14px 11px}
.sm-about summary::-webkit-details-marker{display:none}
.sm-id{display:grid;gap:3px;min-width:0}
.sm-name{display:flex;align-items:baseline;gap:8px;font:700 15px/1.15 ${SANS};color:${BRAND.text};letter-spacing:-.005em}
.sm-ver{font:500 10.5px/1 ${MONO};color:${BRAND.muted};border:1px solid ${BRAND.line};border-radius:5px;padding:2px 5px}
.sm-by{font:12px/1.3 ${SANS};color:${BRAND.muted}}.sm-by b{color:${BRAND.text};font-weight:600}
.sm-more{margin-left:auto;flex:0 0 auto;height:30px;box-sizing:border-box;display:inline-flex;align-items:center;gap:8px;font:600 12px/1 ${SANS};color:${BRAND.text};background:${BRAND.navy2};border:1px solid ${BRAND.line};border-radius:8px;padding:0 12px}
.sm-about summary:hover .sm-more{background:${BRAND.navy3};border-color:#4a4a80}
.sm-more::after{content:"";width:6px;height:6px;border-right:1.5px solid currentColor;border-bottom:1.5px solid currentColor;transform:translateY(-2px) rotate(45deg);transition:transform .15s}
.sm-about[open] .sm-more::after{transform:translateY(1px) rotate(-135deg)}
.sm-motif{display:inline-flex;align-items:center;gap:0;flex:0 0 auto}
.sm-motif i{width:8px;height:8px;border-radius:50%;display:block}.sm-motif s{width:12px;border-top:1.5px dashed ${BRAND.muted};display:block;opacity:.7}
.sm-about-body{display:grid;gap:10px;padding:2px 14px 14px;font:12.5px/1.5 ${SANS};color:#c6c6de;max-width:72ch}
.sm-about-body p{margin:0}
.sm-about-links{display:flex;gap:8px;flex-wrap:wrap}
.sm-about-links a{text-decoration:none}
.sm-about b{color:${BRAND.text}}
@media (prefers-reduced-motion:reduce){.sm-more::after{transition:none}}
`;

function injectCss() {
    if (document.getElementById("sm-css")) return;
    const s = document.createElement("style");
    s.id = "sm-css";
    s.textContent = css;
    document.head.appendChild(s);
}

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

// 24x24 line icons (stroke = currentColor), so buttons are told apart by symbol, not by colour.
const ICON = {
    plate: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 15l5-5 4 4 3-3 6 6"/>',
    find: '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
    preview: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.8"/>',
    run: '<path d="M7 4.5v15l12-7.5z"/>',
    clear: '<path d="M6 6l12 12M18 6L6 18"/>',
    reload: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
    tune: '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
    next: '<circle cx="9" cy="8" r="3.5"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6M19 8v6M16 11h6"/>',
};

/** filename_prefix for the next subject: shot -> shot_2 -> shot_3 (its own folder, the last subject's files stay). */
const nextPrefix = (p) => {
    const m = /^(.*)_(\d+)$/.exec(p);
    return m ? `${m[1]}_${+m[2] + 1}` : `${p}_2`;
};

const SAM3_MISSING = "Needs the ComfyUI-SAM3 node pack: open the Manager > Custom Nodes Manager, search SAM3, install " +
    "ComfyUI-SAM3, then restart ComfyUI.";
/** True when the SAM3 nodes are not installed (picking and detecting need them). */
const noSam3 = () => !LiteGraph.registered_node_types?.LoadSAM3Model;

/** Button with an icon and a label span (change b.label.textContent, not b.textContent, to keep the icon). */
function btn(cls, icon, text) {
    const b = el("button", `sm-btn ${cls}`.trim());
    b.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${ICON[icon]}</svg>`;
    b.label = el("span", null, text);
    b.append(b.label);
    return b;
}

const viewUrl = (f) => api.apiURL(`/view?filename=${encodeURIComponent(f.filename)}&subfolder=${encodeURIComponent(f.subfolder)}&type=${f.type}&r=${Math.random()}`);

function loadImage(src) {
    return new Promise((res, rej) => {
        const im = new Image();
        im.onload = () => res(im);
        im.onerror = rej;
        im.src = src;
    });
}

/** Tint a white-on-black mask PNG into a coloured overlay (offscreen canvas) and keep pixels for hit tests. */
function tintMask(img, color) {
    const c = document.createElement("canvas");
    c.width = img.width; c.height = img.height;
    const g = c.getContext("2d");
    g.drawImage(img, 0, 0);
    const data = g.getImageData(0, 0, c.width, c.height);
    const [r, gg, b] = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16));
    const hit = new Uint8Array(c.width * c.height);
    for (let i = 0; i < hit.length; i++) {
        const v = data.data[i * 4];
        hit[i] = v > 127 ? 1 : 0;
        data.data[i * 4] = r; data.data[i * 4 + 1] = gg; data.data[i * 4 + 2] = b;
        data.data[i * 4 + 3] = v > 127 ? 255 : 0;
    }
    g.putImageData(data, 0, 0);
    return { canvas: c, hit, w: c.width, h: c.height };
}

function hideWidget(w) {
    if (!w || w.sm_hidden) return;
    w.sm_hidden = { type: w.type, computeSize: w.computeSize };
    w.type = "sm-hidden";
    w.hidden = true;
    w.computeSize = () => [0, -4];
}

function showWidget(w) {
    if (!w || !w.sm_hidden) return;
    w.type = w.sm_hidden.type;
    w.computeSize = w.sm_hidden.computeSize;
    w.hidden = false;
    delete w.sm_hidden;
}

/** Queue a tiny helper graph (detect / part preview) and resolve with the ui payload of its report node. */
async function runHelper(prompt, reportId, key, onWait) {
    // Listen BEFORE posting: a fully cached helper finishes in ~10 ms and its websocket "executed" message can arrive
    // before the /prompt response that tells us the prompt id. Events are buffered until the id is known, and
    // /history is polled as a backstop in case a message is missed anyway (reconnect, other tab).
    const seen = [];
    let pid = null, settle = null;
    const onExec = (e) => { seen.push(["ok", e.detail]); settle?.(); };
    const onErr = (e) => { seen.push(["err", e.detail]); settle?.(); };
    api.addEventListener("executed", onExec);
    api.addEventListener("execution_error", onErr);
    const done = () => { api.removeEventListener("executed", onExec); api.removeEventListener("execution_error", onErr); };
    try {
        const res = await api.fetchApi("/prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, client_id: api.clientId, front: true }),
        });
        const body = await res.json();
        if (!res.ok || body.error) throw new Error(body.error?.message || JSON.stringify(body.node_errors || body));
        pid = body.prompt_id;
    } catch (e) {
        done();
        throw e;
    }
    return new Promise((resolve, reject) => {
        let timer = null;
        const finish = (fn, v) => { done(); clearInterval(timer); settle = null; onWait?.(false); fn(v); };
        settle = () => {
            for (const [kind, d] of seen) {
                if (d?.prompt_id !== pid) continue;
                if (kind === "err") return finish(reject, new Error(d.exception_message || "helper failed"));
                if (d.node === reportId && d.output?.[key]) return finish(resolve, d.output[key][0]);
            }
        };
        settle();
        timer = setInterval(async () => {
            try {
                const h = await (await api.fetchApi(`/history/${pid}`)).json();
                const entry = h[pid];
                if (!settle) return;
                if (!entry) {  // not finished: is it waiting behind someone else's render? (ComfyUI runs one job at a time)
                    const q = await (await api.fetchApi("/queue")).json();
                    const running = (q.queue_running || []).map((x) => x[1]);
                    onWait?.(running.length > 0 && !running.includes(pid));
                    return;
                }
                const out = entry?.outputs?.[reportId]?.[key];
                if (out) return finish(resolve, out[0]);
                if (entry.status?.status_str === "error") finish(reject, new Error("helper failed (see the ComfyUI console)"));
            } catch { /* server busy: try again */ }
        }, 1000);
    });
}

/** A point well inside a mask overlay, preferring its centroid, away from (x,y). Normalized coords or null. */
function interiorPoint(o, x, y) {
    if (!o) return null;
    let sx = 0, sy = 0, n = 0;
    for (let j = 0; j < o.h; j += 4) for (let i = 0; i < o.w; i += 4) if (o.hit[j * o.w + i]) { sx += i; sy += j; n++; }
    if (!n) return null;
    let cx = sx / n, cy = sy / n;
    if (!o.hit[Math.round(cy) * o.w + Math.round(cx)]) {  // concave mask: nearest inside pixel to the centroid
        let best = Infinity, bx = cx, by = cy;
        for (let j = 0; j < o.h; j += 4) for (let i = 0; i < o.w; i += 4) {
            if (!o.hit[j * o.w + i]) continue;
            const d = (i - cx) ** 2 + (j - cy) ** 2;
            if (d < best) { best = d; bx = i; by = j; }
        }
        cx = bx; cy = by;
    }
    const nx = cx / o.w, ny = cy / o.h;
    return Math.hypot(nx - x, ny - y) < 0.03 ? null : [nx, ny];
}

/** Share of mask b covered by mask a (both overlays of the same size). */
function coverage(a, b) {
    if (!a || !b || a.w !== b.w || a.h !== b.h) return 0;
    let inter = 0, area = 0;
    for (let i = 0; i < b.hit.length; i += 3) if (b.hit[i]) { area++; if (a.hit[i]) inter++; }
    return area ? inter / area : 0;
}

/** Collapsible "About" section at the bottom of the node: what it does, who made it, the license. */
function aboutPanel() {
    const link = (href, text) => {
        const a = el("a", "sm-btn", text);
        a.href = href; a.target = "_blank"; a.rel = "noopener";
        return a;
    };
    const para = (...parts) => {
        const p = el("p");
        p.append(...parts.map((x) => (typeof x === "string" ? document.createTextNode(x) : x)));
        return p;
    };
    const details = el("details", "sm-about");
    const sum = el("summary");
    const motif = el("span", "sm-motif");
    [BRAND.blue, BRAND.purple, BRAND.amber].forEach((c, i) => {
        if (i) motif.append(el("s"));
        const d = el("i"); d.style.background = c; motif.append(d);
    });
    const id = el("div", "sm-id");
    const name = el("div", "sm-name", TITLE);
    name.append(el("span", "sm-ver", `v${VERSION}`));
    id.append(name);
    sum.append(motif, id, el("span", "sm-more", "About"));
    details.append(sum);
    const body = el("div", "sm-about-body");
    const links = el("div", "sm-about-links");
    links.append(
        link("https://comfycompositing.com", "comfycompositing.com"),
        link("https://comfycompositing.com/contact", "Commercial license"),
        link("https://github.com/rik-python/ComfyUI-Comfy-Pro-Alpha-Generator", "Docs and updates"),
    );
    const by = el("p", "sm-by");
    by.append("Created by ", el("b", null, "Rikin Patel"), " at Comfy Compositing.");
    body.append(
        by,
        para("Mattes one chosen subject with generative edge quality. SAM3 picks the subject, LTX-2.5 alpha-gen draws the edge."),
        para("Free for personal and non-commercial use (PolyForm Noncommercial 1.0.0). Commercial use needs a license from Comfy Compositing."),
        links,
    );
    details.append(body);
    for (const ev of ["pointerdown", "mousedown"]) details.addEventListener(ev, (e) => e.stopPropagation());  // not a canvas drag
    return details;
}

class Picker {
    constructor(node) {
        this.node = node;
        this.w = Object.fromEntries(node.widgets.map((w) => [w.name, w]));
        this.info = null;          // plate proxy: {frames, width, height, fingerprint, subfolder, first_frame}
        this.frame = 0;
        this.img = null;           // current plate frame
        this.subjects = [];          // detections on picks.detect.frame: {id, box, area, mask, overlay}
        this.parts = {};           // token -> overlay
        this.result = null;        // {info, previews, view}
        this.pending = null;       // Final's report while edge colour is still running
        this.tab = "pick";
        this.wipe = 0.5;
        this.picks = this.readPicks();
        this.build();
        this.restore();
        // a run that fails or is cancelled during edge colour must not stay on "still working"
        const stop = () => { if (this.pending) { this.pending = null; this.render(); } };
        api.addEventListener("execution_error", stop);
        api.addEventListener("execution_interrupted", stop);
    }

    readPicks() {
        try { return Object.assign({ cuts: [], adds: [], others: [] }, JSON.parse(this.w.picks.value || "{}")); }
        catch { return { cuts: [], adds: [], others: [] }; }
    }

    writePicks() {
        this.w.picks.value = JSON.stringify(this.picks);
        this.node.setDirtyCanvas(true, true);
        this.updateStatus();
    }

    // ------------------------------------------------------------ DOM
    build() {
        injectCss();
        const root = el("div", "sm-root");

        // Guided layout: a first-time user sees only the next action. Step tracker on top (Plate > Subject > Render),
        // the viewer explains the current step (big Load plate button when empty, "click the subject" after), and
        // one action bar at the bottom. Pick/Result and the search box appear only when they make sense (syncUI).
        this.steps = {};
        const stepper = el("div", "sm-stepper");
        [["plate", "Plate"], ["subject", "Subject"], ["render", "Render"]].forEach(([k, label], i) => {
            if (i) stepper.append(el("i", "sm-st-line"));
            const s = el("div", "sm-st");
            s.append(el("span", "sm-st-dot", String(i + 1)), el("span", null, label));
            this.steps[k] = s;
            stepper.append(s);
        });
        this.reloadBtn = btn("quiet icon", "reload", "");
        this.reloadBtn.title = "Load the plate again (after changing the clip, crop or resize upstream)";
        this.reloadBtn.onclick = () => this.loadPlate();
        this.advBtn = btn("quiet", "tune", "Advanced");
        this.advBtn.onclick = () => this.toggleAdvanced();
        stepper.append(el("span", "sm-sp"), this.reloadBtn, this.advBtn);

        this.tabs = el("div", "sm-seg sm-ov-tl");
        for (const [k, label] of [["pick", "Pick"], ["result", "Result"]]) {
            const t = el("button", null, label);
            t.onclick = () => { this.tab = k; this.seek(this.frame); };
            t.dataset.k = k;
            this.tabs.appendChild(t);
        }
        this.loadBtn = btn("primary big", "plate", "Load plate");
        this.loadBtn.title = "Run the nodes feeding 'images' and show the plate here";
        this.loadBtn.onclick = () => this.loadPlate();
        this.emptyCard = el("div", "sm-empty");
        this.emptyTitle = el("div", "sm-empty-title", "Step 1: show the plate");
        this.emptyText = el("div", "sm-empty-text");
        this.emptyCard.append(this.emptyTitle, this.loadBtn, this.emptyText);
        this.promptBox = el("input", "sm-input");
        this.promptBox.placeholder = "person, dog, bike…";
        this.promptBox.title = "What to detect. Comma-separate several (e.g. 'dog, person' to also exclude people)";
        this.promptBox.value = this.picks.prompt || this.picks.detect?.prompt || "person";
        // Keys typed here must not reach the graph canvas (Delete, Ctrl+C, space...).
        for (const ev of ["keydown", "keyup", "keypress"]) this.promptBox.addEventListener(ev, (e) => e.stopPropagation());
        this.promptBox.addEventListener("keydown", (e) => { if (e.key === "Enter") this.detect(); });
        this.promptBox.addEventListener("change", () => { this.picks.prompt = this.promptText(); this.writePicks(); });
        this.detectBtn = btn("", "find", "Find");
        this.detectBtn.title = "Detect every match on this frame, then click the one you want";
        this.detectBtn.onclick = () => this.detect();
        this.findEl = el("div", "sm-find sm-ov-tr");
        this.findEl.append(this.promptBox, this.detectBtn);

        this.clearBtn = btn("quiet", "clear", "Clear picks");
        this.clearBtn.onclick = () => {
            this.picks = { cuts: [], adds: [], others: [], detect: this.picks.detect, prompt: this.picks.prompt };
            this.writePicks(); this.render();
        };
        this.newBtn = btn("quiet", "next", "New subject");
        this.newBtn.title = "Done with this subject? Keeps its files, clears the picks, and renders the next subject " +
            "(another person on the same plate) into its own folder (filename_prefix _2, _3...)";
        this.newBtn.onclick = () => this.newSubject();
        this.previewBtn = btn("", "preview", "Preview 49 frames");
        this.previewBtn.title = "Matte 49 frames from the pick frame (a few minutes) to check the edges first";
        this.previewBtn.onclick = () => this.queue(true);
        this.runBtn = btn("", "run", "Render full shot");
        this.runBtn.title = "Matte every frame from the pick frame to the end and save the files";
        this.runBtn.onclick = () => this.queue(false);
        this.actions = el("div", "sm-actions");
        this.actions.append(this.clearBtn, this.newBtn, el("span", "sm-sp"), this.runBtn, this.previewBtn);

        const view = el("div", "sm-view");
        this.canvas = el("canvas");
        this.hint = el("div", "sm-hint");
        view.append(this.canvas, this.hint, this.tabs, this.findEl, this.emptyCard);
        this.canvas.addEventListener("pointerdown", (e) => this.onPointer(e));
        this.canvas.addEventListener("pointermove", (e) => { if (e.buttons && this.tab === "result") this.onPointer(e); });
        this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
        new ResizeObserver(() => this.render()).observe(view);

        this.scrub = el("div", "sm-scrub");
        this.slider = el("input");
        this.slider.type = "range"; this.slider.min = 0; this.slider.value = 0;
        this.slider.oninput = () => this.seek(+this.slider.value);
        this.frameLabel = el("span", null, "frame 1");
        this.scrub.append(this.slider, this.frameLabel);

        this.status = el("div", "sm-bar");
        this.qc = el("div", "sm-qc");
        root.append(stepper, view, this.scrub, this.status, this.qc, this.actions, aboutPanel());

        // Min height only, no max: the frontend gives the panel all spare node height, so it grows/shrinks when the
        // node is resized. (Never derive the height from node.size ourselves: the node grows to fit -> feedback loop.)
        this.widget = this.node.addDOMWidget("pro_alpha_ui", "sm-ui", root, {
            serialize: false, hideOnZoom: false, getMinHeight: () => 420,
        });
        // The frontend's side panel (Parameters / app mode) draws a copy of each widget and writes ITS width into
        // widget.width; the canvas then sizes our panel to that (~265 px) instead of the node. Always use node width.
        Object.defineProperty(this.widget, "width", { configurable: true, get: () => undefined, set: () => {} });

        HIDDEN.forEach((n) => hideWidget(this.w[n]));
        ADVANCED.forEach((n) => hideWidget(this.w[n]));
        for (const name of ["overlap_mode", "hair_guide", "edge_colour"]) {
            const w = this.w[name];
            if (!w) continue;
            const cb = w.callback;
            w.callback = (...a) => { const r = cb?.apply(w, a); this.syncQuality(); return r; };
        }
        this.syncQuality();
        this.updateStatus();
    }

    /** Overlap (v13) replaces the quality setting. Every control stays VISIBLE (hidden ones get missed) and is
     *  greyed out while it has no effect: quality in v13, the v13 options without the switch. */
    syncQuality() {
        const v13 = !!this.w.overlap_mode?.value;
        const advanced = !this.w[ADVANCED[0]]?.sm_hidden;
        const grey = (w, off) => { if (w) w.disabled = off; };
        grey(this.w.quality, v13);
        grey(this.w.v13_depth, !v13);
        grey(this.w.hair_guide, !v13);
        grey(this.w.hair_radius, !(v13 && this.w.hair_guide?.value));
        V13_ADVANCED.forEach((n) => {
            (advanced ? showWidget : hideWidget)(this.w[n]);
            grey(this.w[n], !v13);
        });
        // v14: only plates bigger than 1920x1088 have native-resolution edges to render (unknown size: active)
        const iw = this.info?.width, ih = this.info?.height;
        const [bw, bh] = ih > iw ? [1088, 1920] : [1920, 1088];  // LTX's frame in the plate's orientation
        const big = !iw || iw > bw || ih > bh;
        grey(this.w.native_edges, !big);
        grey(this.w.clean_plate_lora, this.w.edge_colour?.value !== "perfect");  // v15: only perfect uses it
        this.fitNode();
    }

    /** Plate frame number of index i (first_frame, e.g. 1001 for VFX plates). */
    num(i) { return (this.info?.first_frame ?? this.w.first_frame?.value ?? 1) + i; }

    /** API prompt of everything feeding our `images` (and first_frame) input, plus that link. */
    async upstream() {
        const { output } = await app.graphToPrompt();
        const me = output[String(this.node.id)];
        if (!me) throw new Error(`The ${TITLE} node is muted or bypassed.`);
        const images = me.inputs.images;
        if (!Array.isArray(images)) throw new Error("Connect the plate to the 'images' input first.");
        const prompt = {};
        const visit = (id) => {
            if (prompt[id] || !output[id]) return;
            prompt[id] = output[id];
            for (const v of Object.values(output[id].inputs || {})) if (Array.isArray(v) && typeof v[0] === "string") visit(v[0]);
        };
        visit(images[0]);
        const ff = me.inputs.first_frame;
        if (Array.isArray(ff)) visit(ff[0]);
        return { prompt, images, firstFrame: ff ?? 1 };
    }

    /** Run a helper graph on the CURRENT plate (upstream nodes are cached by ComfyUI, so this is quick). */
    async helper(build, reportId, key) {
        const up = await this.upstream();
        const onWait = (w) => { if (w !== !!this.waiting) { this.waiting = w; this.render(); } };
        return runHelper({ ...up.prompt, ...build(up.images, up.firstFrame) }, reportId, key, onWait);
    }

    async loadPlate() {
        this.loadBtn.disabled = true;
        this.loadBtn.label.textContent = "Loading…";
        try {
            const info = await this.helper((images, firstFrame) => ({
                pp: { class_type: "ComfyProAlpha_PlateProxy", inputs: { images, width: 960, first_frame: firstFrame } },
            }), "pp", "comfy_pro_alpha_plate");
            const changed = this.picks.view && this.picks.view.fingerprint !== info.fingerprint;
            this.info = info;
            this.syncQuality();  // native_edges greys out for plates of 1920x1088 or smaller
            this.picks.view = info;
            if (changed && this.picks.target) {
                this.picks.stale = true;  // plate changed (crop/resize/new clip) after picking
            } else if (changed) {
                this.picks = { cuts: [], adds: [], others: [], view: info };
                this.subjects = []; this.parts = {};
            }
            this.writePicks();
            this.slider.max = info.frames - 1;
            this.tab = "pick";
            this.seek(Math.min(this.frame, info.frames - 1));
        } catch (e) {
            alert(`${TITLE}: could not load the plate\n${e.message}`);
        } finally {
            this.loadBtn.disabled = false;
            this.loadBtn.label.textContent = "Load plate";
        }
    }

    /** Grow the node to fit its widgets; never shrink below the size the user dragged it to. */
    fitNode() {
        const min = this.node.computeSize();
        this.node.setSize([Math.max(this.node.size[0], 640), Math.max(this.node.size[1], min[1])]);
        app.graph?.setDirtyCanvas(true, true);
    }

    toggleAdvanced() {
        const show = !!this.w[ADVANCED[0]].sm_hidden;
        ADVANCED.forEach((n) => (show ? showWidget : hideWidget)(this.w[n]));
        this.advBtn.label.textContent = show ? "Hide advanced" : "Advanced";
        this.syncQuality();
    }

    updateStatus() {
        const p = this.picks;
        const chip = (color, text) => {
            const c = el("span", "sm-chip");
            const d = el("span", "sm-dot"); d.style.background = color;
            c.append(d, document.createTextNode(text));
            return c;
        };
        const kids = [];
        if (this.info) {  // LTX alpha-gen is tuned for 1920x1088; a 1920x1080 plate is padded to it automatically
            const { width: w, height: h } = this.info;
            const best = w === 1920 && (h === 1080 || h === 1088);
            const c = chip(best ? ADD : BRAND.amber, best ? `plate ${w}×${h}` : `plate ${w}×${h} (best: 1920×1088)`);
            c.title = best ? "Best size for LTX alpha-gen."
                : "LTX alpha-gen is tuned for 1920x1088 (1080p plates are padded automatically). Other sizes work, "
                  + "but edges can be softer. Resize upstream for best quality; for 4K, matte a 1080p proxy.";
            kids.push(c);
        }
        if (p.target) kids.unshift(chip(TARGET, p.target.box ? `target #${p.target.id + 1} (frame ${this.num(p.target.frame)})`
            : `target: clicked object (frame ${this.num(p.target.frame)})`));
        if (p.others.length) kids.push(chip(COLORS[0], `excluded ${p.others.length}`));
        if (p.cuts.length) kids.push(chip(CUT, `cuts ${p.cuts.length}`));
        if (p.adds.length) kids.push(chip(ADD, `adds ${p.adds.length}`));
        this.status.replaceChildren(...kids);
        this.syncUI();
    }

    /** Next subject on the same plate: keep the last one's files (its own folder), clear the picks and the result.
     *  The plate and this frame's detections stay, so the next subject is one click away. */
    newSubject() {
        const w = this.w.filename_prefix;
        const next = nextPrefix(w.value);
        const kept = this.result?.info?.out_dir || w.value;
        if (!confirm(`Done with this subject?\n\nIts files stay in ${kept}.\nThe picks are cleared and the next ` +
                     `subject renders into ${next}.`)) return;
        w.value = next;
        this.picks = { cuts: [], adds: [], others: [], detect: this.picks.detect, prompt: this.picks.prompt,
                       view: this.picks.view };
        this.parts = {};
        this.result = null;
        this.pending = null;
        this.tab = "pick";
        this.writePicks();
        this.node.setDirtyCanvas?.(true, true);
        this.render();
    }

    /** Show only what the current step needs (first-time users see one next action, never the whole toolbar). */
    syncUI() {
        const plate = !!(this.info && this.img), subject = !!this.picks.target, result = !!this.result;
        const now = !plate ? "plate" : !subject ? "subject" : "render";
        for (const [k, s] of Object.entries(this.steps)) {
            const done = k === "plate" ? plate : k === "subject" ? subject : result;
            s.classList.toggle("done", done && k !== now);
            s.classList.toggle("now", k === now);
        }
        const linked = this.node.inputs?.find((i) => i.name === "images")?.link != null;
        this.emptyCard.hidden = plate || this.tab === "result";
        this.emptyText.textContent = linked
            ? "Runs the nodes feeding 'images' and shows the plate here."
            : "First connect a video to the 'images' input (Load Video, or Load Plate Sequence for EXRs).";
        this.hint.hidden = !this.emptyCard.hidden;
        this.findEl.hidden = !plate || this.tab !== "pick";
        this.tabs.hidden = !result;
        for (const t of this.tabs.children) t.classList.toggle("on", t.dataset.k === this.tab);
        this.reloadBtn.hidden = !this.info;
        this.scrub.hidden = this.status.hidden = !this.info;
        this.actions.hidden = !plate;
        this.clearBtn.hidden = !subject && !this.picks.cuts.length && !this.picks.adds.length;
        this.newBtn.hidden = !result;  // once a subject has rendered
        this.previewBtn.disabled = this.runBtn.disabled = !subject;
        // Preview first; once a preview has rendered, the full shot becomes the main action.
        const previewed = result && /_preview$/.test(this.result.info.out_dir || "");
        this.previewBtn.classList.toggle("primary", !previewed);
        this.runBtn.classList.toggle("primary", previewed);
    }

    // ------------------------------------------------------------ data
    async restore() {
        this.info = this.picks.view || null;
        this.syncQuality();  // reopened workflow: grey native_edges for the saved plate size
        if (!this.info) return this.render();
        this.slider.max = this.info.frames - 1;
        await this.restoreDetections();
        this.seek(this.picks.target?.frame ?? this.picks.detect?.frame ?? 0);
    }

    async restoreDetections() {
        const d = this.picks.detect;
        this.subjects = [];
        if (!d) return this.restoreParts();
        for (const saved of d.subjects || d.people || []) {  // `people`: picks saved before any-subject detection
            const p = { ...saved };  // overlays stay out of the picks JSON (pixel arrays would bloat the workflow)
            try { p.overlay = tintMask(await loadImage(viewUrl(p.mask)), COLORS[p.id % COLORS.length]); }
            catch { p.overlay = null; }  // temp files are gone after a restart: fall back to boxes
            this.subjects.push(p);
        }
        this.frameSubjects = { [d.frame]: this.subjects };
        await this.restoreParts();
    }

    /** Mask overlays of a clicked target and of cuts/adds (temp files: gone after a ComfyUI restart). */
    async restoreParts() {
        const t = this.picks.target;
        if (t && !t.box && t.mask) {
            try { this.parts[t.token] = tintMask(await loadImage(viewUrl(t.mask)), TARGET); } catch { /* gone */ }
        }
        for (const [kind, color] of [["cuts", CUT], ["adds", ADD]]) {
            for (const part of this.picks[kind]) {
                if (!part.mask) continue;
                try { this.parts[part.token] = tintMask(await loadImage(viewUrl(part.mask)), color); }
                catch { /* gone */ }
            }
        }
    }

    async seek(f) {
        if (!this.info) return this.render();
        this.frame = Math.max(0, Math.min(f, this.info.frames - 1));
        this.slider.value = this.frame;
        this.frameLabel.textContent = `frame ${this.num(this.frame)}  (${this.frame + 1} / ${this.info.frames})`;
        if (this.tab === "pick") {
            const want = this.frame;
            try {
                const name = `f${String(want).padStart(5, "0")}.jpg`;
                const im = await loadImage(api.apiURL(`/view?filename=${name}&subfolder=${encodeURIComponent(this.info.subfolder)}&type=temp`));
                if (want === this.frame) this.img = im;
            } catch (e) {
                this.img = null;  // temp proxies are cleared on restart
                this.hint.textContent = "Click Load plate to show the plate.";
            }
        }
        this.render();
    }

    /** The detect prompt as typed, normalised ("dog,  person" -> "dog, person"); "person" when empty. */
    promptText() {
        const terms = this.promptBox.value.split(",").map((t) => t.trim()).filter(Boolean);
        return terms.length ? terms.join(", ") : "person";
    }

    /** SAM3 text detections on one frame, one pass per comma-separated term (masks as overlays). Does not touch picks. */
    async detectOn(frame, prompt) {
        const found = [];
        for (const term of prompt.split(",").map((t) => t.trim()).filter(Boolean)) {
            const token = Math.random().toString(36).slice(2, 10);
            const out = await this.helper((images) => ({
                f: { class_type: "ComfyProAlpha_Frame", inputs: { images, index: frame } },
                m: { class_type: "LoadSAM3Model", inputs: { precision: "auto" } },
                g: { class_type: "SAM3Grounding", inputs: { sam3_model_config: ["m", 0], image: ["f", 0], confidence_threshold: 0.3,
                                                           text_prompt: term, max_detections: -1 } },
                r: { class_type: "ComfyProAlpha_DetectReport", inputs: { masks: ["g", 0], frame, token } },
            }), "r", "comfy_pro_alpha_detect");
            found.push(...(out.subjects || out.people || []).map((p) => ({ ...p, label: term })));
        }
        const subjects = found.sort((a, b) => a.box[0] - b.box[0]).map((p, i) => ({ ...p, id: i }));
        for (const p of subjects) p.overlay = tintMask(await loadImage(viewUrl(p.mask)), COLORS[p.id % COLORS.length]);
        return subjects;
    }

    /** quiet: auto-detect on a click (no alert when nothing is found). */
    async detect(quiet = false) {
        if (noSam3()) { alert(`${TITLE}: ${SAM3_MISSING}`); return; }
        this.detectBtn.disabled = true;
        this.detectBtn.label.textContent = "Finding…";
        try {
            if (!this.info) await this.loadPlate();
            if (!this.info) return;
            const prompt = this.promptText();
            this.promptBox.value = prompt;
            const subjects = await this.detectOn(this.frame, prompt);
            if (!subjects.length) {
                if (!quiet) alert(`${TITLE}: no "${prompt}" found on this frame. Try another word or frame.`);
                return;
            }
            this.frameSubjects = { [this.frame]: subjects };
            this.picks = { cuts: this.picks.cuts, adds: this.picks.adds, others: [], view: this.info, prompt,
                           detect: { frame: this.frame, prompt, subjects: subjects.map(({ overlay, ...p }) => p) } };
            this.subjects = subjects;
            this.writePicks();
            this.tab = "pick";
            this.render();
        } catch (e) {
            alert(`${TITLE}: detection failed\n${e.message}`);
        } finally {
            this.detectBtn.disabled = false;
            this.detectBtn.label.textContent = "Find";
        }
    }

    /** Detections on `frame` (pick-frame prompt) plus which one is the target (nearest to the target box). */
    async subjectsOn(frame) {
        this.frameSubjects = this.frameSubjects || {};
        if (frame === this.picks.detect?.frame && this.subjects.length) this.frameSubjects[frame] = this.subjects;
        if (!this.frameSubjects[frame]) this.frameSubjects[frame] = await this.detectOn(frame, this.picks.detect?.prompt || "person");
        const subjects = this.frameSubjects[frame];
        const t = this.picks.target;
        let target = null;
        if (t && !t.box) {  // clicked target (no detection): its own mask, only on its frame
            target = frame === t.frame && this.parts[t.token] ? { overlay: this.parts[t.token] } : null;
        } else if (t) {
            if (frame === t.frame) target = subjects.find((p) => p.id === t.id) || null;
            if (!target) {
                const cost = (p) => Math.hypot(p.box[0] - t.box[0], p.box[1] - t.box[1]) + Math.abs(p.box[3] - t.box[3]);
                target = [...subjects].sort((a, b) => cost(a) - cost(b))[0] || null;
            }
        }
        return { subjects, target };
    }

    /** SAM3 point segmentation of one click on one frame (single-frame mask preview). Resolves {mask}. */
    segmentPoint(frame, x, y, neg, token) {
        if (noSam3()) return Promise.reject(new Error(SAM3_MISSING));
        const n = {
            f: { class_type: "ComfyProAlpha_Frame", inputs: { images: null, index: frame } },
            m: { class_type: "LoadSAM3Model", inputs: { precision: "auto" } },
            p1: { class_type: "SAM3CreatePoint", inputs: { x, y, is_foreground: true } },
            c: { class_type: "SAM3CombinePoints", inputs: { point_1: ["p1", 0] } },
            s: { class_type: "SAM3Segmentation", inputs: { sam3_model_config: ["m", 0], image: ["f", 0], positive_points: ["c", 0] } },
            r: { class_type: "ComfyProAlpha_PartReport", inputs: { mask: ["s", 0], token } },
        };
        if (neg.length) {
            const negInputs = {};
            neg.forEach(([nx, ny], i) => {
                n[`n${i}`] = { class_type: "SAM3CreatePoint", inputs: { x: nx, y: ny, is_foreground: false } };
                negInputs[`point_${i + 1}`] = [`n${i}`, 0];
            });
            n.cn = { class_type: "SAM3CombinePoints", inputs: negInputs };
            n.s.inputs.negative_points = ["cn", 0];
        }
        return this.helper((images) => {
            n.f.inputs.images = images;
            return n;
        }, "r", "comfy_pro_alpha_part");
    }

    async addPart(kind, x, y) {
        const token = Math.random().toString(36).slice(2, 10);
        const part = { frame: this.frame, x, y, token, points: [[x, y]], neg: [] };
        this.picks[kind].push(part);
        this.writePicks();
        this.render();
        try {
            // Auto negative points so one click gives a PART, never a whole subject:
            //  add -> "not the neighbours" (centres of other detections near the click)
            //  cut -> "not the subject" (centre of the target)
            const { subjects, target } = await this.subjectsOn(part.frame);
            const near = (p) => {
                const [cx, cy, w, h] = p.box;
                return Math.abs(x - cx) < w / 2 + 0.05 && Math.abs(y - cy) < h / 2 + 0.05;
            };
            const negSubjects = kind === "adds" ? subjects.filter((p) => p !== target && near(p)) : (target ? [target] : []);
            part.neg = negSubjects.map((p) => interiorPoint(p.overlay, x, y)).filter(Boolean);

            const out = await this.segmentPoint(part.frame, x, y, part.neg, token);
            part.mask = out.mask;
            const ov = tintMask(await loadImage(viewUrl(out.mask)), kind === "cuts" ? CUT : ADD);
            this.parts[token] = ov;
            // Safety: an add must not swallow another detection, a cut must not swallow the subject.
            const victims = kind === "adds" ? subjects.filter((p) => p !== target) : (target ? [target] : []);
            const bad = victims.find((p) => coverage(ov, p.overlay) > 0.4);
            part.warning = bad ? (kind === "adds"
                ? `This add grabbed #${bad.id + 1} ${bad.label || ""}. Ctrl-click it to remove, then click closer to the part.`
                : "This cut grabbed the subject. Ctrl-click it to remove, then click right on the part to cut.") : null;
            this.writePicks();
            this.render();
        } catch (e) {
            console.warn("[ComfyProAlpha] part preview failed", e);
            part.warning = `Part preview failed: ${e.message}`;
            this.render();
        }
    }

    async queue(preview) {
        if (!this.picks.target) { alert(`${TITLE}: click the subject to matte (anything in the plate, or a match after Find).`); return; }
        if (this.picks.stale && !confirm("The plate changed after you picked the subject. Run anyway?")) return;
        const r = this.w.render;
        r.value = preview ? "preview chunk" : "full shot";
        try {
            await app.queuePrompt(0, 1);
            // queuePrompt can return before the graph is serialized (when the frontend is still sending an earlier
            // prompt): wait until it is done, or the reset below would turn a preview into a full shot.
            for (let i = 0; i < 100 && app.processingQueue; i++) await new Promise((res) => setTimeout(res, 100));
        } finally { r.value = "full shot"; }
    }

    // ------------------------------------------------------------ interaction
    /** Physical screen pixels per widget CSS pixel: graph zoom (CSS transform) x display scaling. */
    screenScale() {
        const k = this.canvas.getBoundingClientRect().width / (this.canvas.clientWidth || 1);
        return (k || 1) * (window.devicePixelRatio || 1);
    }

    imageRect() {
        const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
        const iw = this.info?.width || 16, ih = this.info?.height || 9;
        // Never show the plate bigger than its real size on screen (1 plate pixel = at most 1 screen pixel), however
        // large the node or the graph zoom: blown-up frames invite pixel peeping on proxies. Judge edges in comp.
        const s = Math.min(cw / iw, ch / ih, this.info ? 1 / this.screenScale() : Infinity);
        const w = iw * s, h = ih * s;
        return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
    }

    onPointer(e) {
        const r = this.imageRect();
        const b = this.canvas.getBoundingClientRect();
        // The graph canvas zooms DOM widgets with a CSS transform: the on-screen rect is scaled, clientWidth/imageRect
        // are not. Convert the screen offset back to unscaled widget pixels first, or clicks land off at any zoom != 100%.
        const kx = b.width / (this.canvas.clientWidth || 1), ky = b.height / (this.canvas.clientHeight || 1);
        const x = ((e.clientX - b.left) / kx - r.x) / r.w, y = ((e.clientY - b.top) / ky - r.y) / r.h;
        if (this.tab === "result") { this.wipe = Math.max(0, Math.min(1, x)); this.render(); return; }
        if (x < 0 || x > 1 || y < 0 || y > 1 || e.button !== 0) return;
        e.preventDefault(); e.stopPropagation();
        if (e.ctrlKey || e.metaKey) return this.removeNearest(x, y);
        if (e.altKey) return this.addPart("cuts", x, y);
        if (e.shiftKey) return this.addPart("adds", x, y);
        this.pickSubject(x, y);
    }

    async pickSubject(x, y) {
        if (!this.info) return;
        let d = this.picks.detect;
        if (!d || d.frame !== this.frame) {
            // First click on this frame: detect (box prompt, "person" by default) so the click can land on a detection
            // and everything else detected gets excluded, also when the clicked thing is not a detection.
            await this.detect(true);
            d = this.picks.detect;
            if (!d || d.frame !== this.frame) return this.pickPoint(x, y);
        }
        // Smallest detection whose mask (or box) contains the click: front-most when detections overlap.
        const hits = this.subjects.filter((p) => {
            if (p.overlay) {
                const o = p.overlay;
                return o.hit[Math.floor(y * o.h) * o.w + Math.floor(x * o.w)] === 1;
            }
            const [cx, cy, w, h] = p.box;
            return Math.abs(x - cx) < w / 2 && Math.abs(y - cy) < h / 2;
        }).sort((a, b) => a.area - b.area);
        if (!hits.length) return this.pickPoint(x, y);
        const t = hits[0];
        this.picks.target = { id: t.id, frame: d.frame, box: t.box };
        this.picks.plate = this.info?.fingerprint;
        delete this.picks.stale;
        // Exclude every other detection, not only the ones near the target on this frame: subjects move (a dog that
        // starts far away can run alongside the target later, and then LTX would keep it as attached). Each one
        // costs a SAM3 track, so keep the nearest MAX_OTHERS.
        const [tx, ty] = t.box;
        const dist = (p) => Math.hypot(p.box[0] - tx, p.box[1] - ty);
        this.picks.others = this.subjects.filter((p) => p.id !== t.id).sort((a, b) => dist(a) - dist(b)).slice(0, MAX_OTHERS)
            .map((p) => ({ id: p.id, frame: d.frame, box: p.box }));
        this.writePicks();
        this.render();
    }

    /** Click anywhere that is not a detection: SAM3 segments that point and it becomes the target. */
    async pickPoint(x, y) {
        const token = Math.random().toString(36).slice(2, 10);
        const t = { frame: this.frame, x, y, points: [[x, y]], neg: [], token };
        this.picks.target = t;
        this.picks.plate = this.info?.fingerprint;
        delete this.picks.stale;
        // Detections on this frame are other subjects: exclude them (nearest first), as for a detected target.
        const d = this.picks.detect;
        const dist = (p) => Math.hypot(p.box[0] - x, p.box[1] - y);
        this.picks.others = d && d.frame === this.frame
            ? this.subjects.slice().sort((a, b) => dist(a) - dist(b)).slice(0, MAX_OTHERS)
                .map((p) => ({ id: p.id, frame: d.frame, box: p.box }))
            : [];
        this.writePicks();
        this.render();
        try {
            const out = await this.segmentPoint(t.frame, x, y, [], token);
            t.mask = out.mask;
            this.parts[token] = tintMask(await loadImage(viewUrl(out.mask)), TARGET);
            // A detection the clicked object mostly covers is part of it (a person standing on the clicked rug): keep it.
            const ov = this.parts[token];
            const inside = new Set(this.subjects.filter((p) => coverage(ov, p.overlay) > 0.6).map((p) => p.id));
            this.picks.others = this.picks.others.filter((o) => !inside.has(o.id));
        } catch (e) {
            console.warn("[ComfyProAlpha] target preview failed", e);
            if (noSam3()) alert(`${TITLE}: ${SAM3_MISSING}`);
        }
        this.writePicks();
        this.render();
    }

    removeNearest(x, y) {
        let best = null;
        for (const kind of ["cuts", "adds"]) {
            this.picks[kind].forEach((p, i) => {
                if (p.frame !== this.frame) return;
                const dist = Math.hypot(p.x - x, p.y - y);
                if (dist < 0.03 && (!best || dist < best.dist)) best = { kind, i, dist };
            });
        }
        if (best) { this.picks[best.kind].splice(best.i, 1); this.writePicks(); this.render(); }
    }

    // ------------------------------------------------------------ drawing
    render() {
        const c = this.canvas, dpr = window.devicePixelRatio || 1;
        c.width = Math.max(1, c.clientWidth * dpr); c.height = Math.max(1, c.clientHeight * dpr);
        const g = c.getContext("2d");
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, c.clientWidth, c.clientHeight);
        this.updateStatus();
        if (this.tab === "result") return this.renderResult(g);
        const r = this.imageRect();
        if (this.img) g.drawImage(this.img, r.x, r.y, r.w, r.h);

        const d = this.picks.detect;
        const onDetect = d && d.frame === this.frame;
        if (onDetect) {
            for (const p of this.subjects) {
                const isT = this.picks.target?.id === p.id;
                const color = isT ? TARGET : COLORS[p.id % COLORS.length];
                if (p.overlay) {
                    g.globalAlpha = isT ? 0.55 : 0.28;
                    g.drawImage(isT ? tintCache(p, TARGET) : p.overlay.canvas, r.x, r.y, r.w, r.h);
                    g.globalAlpha = 1;
                }
                const [cx, cy, w, h] = p.box;
                const bx = r.x + (cx - w / 2) * r.w, by = r.y + (cy - h / 2) * r.h;
                g.strokeStyle = color;
                g.lineWidth = isT ? 2.5 : 1;
                g.setLineDash(isT ? [] : [4, 4]);
                g.strokeRect(bx, by, w * r.w, h * r.h);
                g.setLineDash([]);
                const tag = p.label ? `${p.id + 1} ${p.label}` : String(p.id + 1);
                g.font = `600 12px ${SANS}`; g.textAlign = "left"; g.textBaseline = "middle";
                const tw = g.measureText(tag).width + 12;
                g.fillStyle = color;
                g.beginPath(); g.roundRect(bx + 2, by + 2, tw, 20, 10); g.fill();
                g.fillStyle = BRAND.navy;
                g.fillText(tag, bx + 8, by + 12);
            }
        }
        const pt = this.picks.target;
        if (pt && !pt.box && pt.frame === this.frame) {
            const ov = this.parts[pt.token];
            if (ov) { g.globalAlpha = 0.55; g.drawImage(ov.canvas, r.x, r.y, r.w, r.h); g.globalAlpha = 1; }
            const px = r.x + pt.x * r.w, py = r.y + pt.y * r.h;
            g.strokeStyle = TARGET; g.lineWidth = 3;
            g.beginPath(); g.arc(px, py, 9, 0, Math.PI * 2); g.stroke();
            g.fillStyle = TARGET; g.beginPath(); g.arc(px, py, 3, 0, Math.PI * 2); g.fill();
        }
        for (const [kind, color, sym] of [["cuts", CUT, "×"], ["adds", ADD, "+"]]) {
            for (const p of this.picks[kind]) {
                if (p.frame !== this.frame) continue;
                const ov = this.parts[p.token];
                if (ov) { g.globalAlpha = 0.5; g.drawImage(ov.canvas, r.x, r.y, r.w, r.h); g.globalAlpha = 1; }
                const px = r.x + p.x * r.w, py = r.y + p.y * r.h;
                g.fillStyle = color; g.beginPath(); g.arc(px, py, 9, 0, Math.PI * 2); g.fill();
                g.fillStyle = BRAND.navy; g.font = `600 14px ${SANS}`; g.textAlign = "center"; g.textBaseline = "middle";
                g.fillText(sym, px, py + 1);
            }
        }
        if (this.waiting) {
            this.hint.style.background = "#4a3514e6";
            this.hint.textContent = "⏳ Waiting for the running render to finish (ComfyUI runs one job at a time). "
                + "Cancel it in the queue panel to go now.";
            return;
        }
        const warn = [...this.picks.cuts, ...this.picks.adds].find((p) => p.warning);
        this.hint.style.background = warn || this.picks.stale ? "#5a1a2ae6" : "";
        this.hint.textContent = warn ? `⚠ ${warn.warning}`
            : this.picks.stale ? "⚠ The plate changed (crop/resize/new clip) after picking: Load plate, then pick again."
            : !this.info ? "Connect the plate to 'images', then click Load plate."
            : !this.picks.target && !d ? "Step 2: click the subject you want to matte. Hard to click? Type it in Find (top right)."
            : !this.picks.target ? `Detections live on frame ${this.num(d.frame)}. Click the one you want there, or click anything on this frame.`
            : !onDetect && this.picks.target.frame !== this.frame ? "Cuts/adds can go on any frame (place them where the part first appears)."
            : !this.picks.target ? "Click a detection, or anything else, to matte it. Other detections are excluded."
            : "Fix the pick: Alt-click cuts a part, Shift-click adds one (the glass, the table), Ctrl-click removes. Then Preview.";
    }

    renderResult(g) {
        const res = this.result;
        const r = this.imageRect();
        if (this.pending) {
            this.hint.style.background = "#4a3514e6";
            this.hint.textContent = `Matte saved (${this.pending.frames} frames). Still working: building edge colour ` +
                `(${this.w.edge_colour?.value}) on the CPU, see the progress bar. The result shows here when it is done.`;
            return;
        }
        if (!res || !res.previews.length) { this.hint.textContent = "Run Preview 49 frames or Render full shot to see the result here."; return; }
        const items = res.previews;
        const first = res.info.first_frame - (this.info?.first_frame ?? 1);
        // nearest preview to the scrubber frame (first is the processed range's index offset)
        let it = items[0];
        for (const x of items) if (Math.abs(first + x.frame - this.frame) < Math.abs(first + it.frame - this.frame)) it = x;
        const key = res.view === "alpha" ? "alpha" : "comp";
        if (!it._img) {
            it._img = {};
            Promise.all(["plate", "comp", "alpha"].map((k) => loadImage(viewUrl(it[k])).then((im) => (it._img[k] = im))))
                .then(() => this.render()).catch(() => {});
            return;
        }
        const a = it._img.plate, b = it._img[key];
        if (!a || !b) return;
        g.drawImage(a, r.x, r.y, r.w, r.h);
        const split = r.w * this.wipe;
        g.save(); g.beginPath(); g.rect(r.x + split, r.y, r.w - split, r.h); g.clip();
        g.drawImage(b, r.x, r.y, r.w, r.h); g.restore();
        g.strokeStyle = TARGET; g.lineWidth = 2;
        g.beginPath(); g.moveTo(r.x + split, r.y); g.lineTo(r.x + split, r.y + r.h); g.stroke();
        const q = res.info;
        const kind = /_preview$/.test(q.out_dir || "") ? "preview" : "full shot";
        const range = `${kind}: frames ${q.first_frame}-${q.first_frame + q.frames - 1}`;
        const inside = this.frame >= first && this.frame < first + q.frames;
        this.hint.style.background = inside ? "" : "#4a3514e6";
        this.hint.textContent = inside
            ? `${range} · ${q.mode} · frame ${q.first_frame + it.frame} · drag to wipe plate | ${key}` +
              (q.out_dir ? ` · saved: ${q.out_dir}` : "")
            : `Frame ${this.num(this.frame)} is outside this render (${range}). Showing the nearest rendered frame.` +
              (kind === "preview" ? " Render full shot to do the rest." : "");
    }

    showResult(output) {
        const info = output?.pro_alpha?.[0];
        if (!info) return;
        // Final reports first, when the matte is saved. With edge colour on, EdgeColour then runs for minutes on the
        // CPU and reports again (its report has edge_colour): until then say it is still working, not a finished result.
        if (info.edge_colour == null && (this.w.edge_colour?.value ?? "off") !== "off") {
            this.pending = info;
            this.tab = "result";
            this.render();
            return;
        }
        this.pending = null;
        this.result = { info, previews: output.previews || [], view: this.result?.view || "alpha" };
        this.tab = "result";
        const badge = (label, value, cls) => el("span", `sm-badge ${cls}`, `${label} ${value}`);
        const kids = [];
        // Flicker and soft edge are measured against LTX alone: meaningless when LTX barely matted the subject.
        const vsLtx = info.ltx_cover == null || info.ltx_cover >= 60;
        if (!vsLtx) {
            const b = badge("flicker / soft edge", "n/a", "");
            b.title = "Measured against LTX, which did not matte this subject: mostly a ViTMatte result. Judge by eye.";
            kids.push(b);
        } else if (info.flicker_x != null) {
            const f = info.flicker_x;
            kids.push(badge("flicker", `${f.toFixed(2)}× LTX`, f <= 1.3 ? "sm-g" : f <= 2 ? "sm-a" : "sm-r"));
        }
        if (vsLtx && info.detail_pct != null) {
            const d = info.detail_pct;  // edge vs LTX alone: kids benchmark 88.7% (ViTMatte 45%, SAM3 mask 29%)
            const b = badge("detail match", `${d.toFixed(1)}%`, d >= 80 ? "sm-g" : d >= 60 ? "sm-a" : "sm-r");
            b.title = "How closely the edge matches LTX alpha-gen's hair and soft detail (100% = identical).";
            kids.push(b);
        }
        if (info.ltx_cover != null) {
            const c = info.ltx_cover;
            const b = badge("LTX cover", `${c.toFixed(0)}%`, c >= 90 ? "sm-g" : c >= 60 ? "sm-a" : "sm-r");
            b.title = "How much of the subject LTX alpha-gen matted. Low: LTX did not see it as the main subject, "
                    + "so those frames use the ViTMatte edge instead.";
            kids.push(b);
        }
        if (info.native) {  // v14: plate bigger than 1920x1088, matted on a proxy
            const b = badge("native", info.native_edges ? `${info.native} · ${info.tiles} tiles` : `${info.native} · scaled up`,
                            info.native_edges ? "sm-g" : "");
            b.title = info.native_edges
                ? "Matted on a 1080p proxy; the edges were rendered again on native-resolution tiles and merged in."
                : "Matted on a 1080p proxy and scaled up (native-resolution edges off, or no edge found to refine).";
            kids.push(b);
        }
        if (info.edge_colour) {  // v15: clean colour in the soft edges (absent = off, the raw plate)
            const b = badge("edge colour", info.edge_colour, info.edge_note ? "sm-a" : "sm-g");
            b.title = info.edge_note || (info.edge_colour === "perfect"
                ? "Background rebuilt by LTX Clean Plate and subtracted from the edges; rgba_ files saved."
                : "Edge colour rebuilt from the subject itself; rgba_ files saved.");
            kids.push(b);
        }
        const tog = btn("quiet", "preview", this.result.view === "alpha" ? "Show comp" : "Show alpha");
        tog.onclick = () => {
            this.result.view = this.result.view === "alpha" ? "comp" : "alpha";
            tog.label.textContent = this.result.view === "alpha" ? "Show comp" : "Show alpha";
            this.render();
        };
        kids.push(tog);
        this.qc.replaceChildren(...kids);
        this.seek(info.first_frame - (this.info?.first_frame ?? 1));
    }
}

const _tints = new WeakMap();
function tintCache(p, color) {
    let m = _tints.get(p);
    if (!m) { m = {}; _tints.set(p, m); }
    if (!m[color]) {
        const c = document.createElement("canvas");
        c.width = p.overlay.w; c.height = p.overlay.h;
        const g = c.getContext("2d");
        g.drawImage(p.overlay.canvas, 0, 0);
        g.globalCompositeOperation = "source-in";
        g.fillStyle = color; g.fillRect(0, 0, c.width, c.height);
        m[color] = c;
    }
    return m[color];
}

app.registerExtension({
    name: "ComfyProAlpha.Picker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Brand colours for every node of the pack (title bar navy, body a step lighter). A colour the user picks
        // from the node's right-click menu is saved in the workflow and wins on load.
        const BRANDED = [NODE, LEGACY, "ComfyProAlpha_LoadPlateSequence", "SelectiveMatte_LoadPlateSequence"];
        if (BRANDED.includes(nodeData.name)) {
            const made = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = made?.apply(this, arguments);
                this.color = BRAND.navy;
                this.bgcolor = BRAND.navy2;
                return r;
            };
        }
        if (nodeData.name !== NODE && nodeData.name !== LEGACY) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created?.apply(this, arguments);
            this.smPicker = new Picker(this);
            return r;
        };
        const connections = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = connections?.apply(this, arguments);
            this.smPicker?.syncUI();  // the empty viewer says "connect a video" until 'images' is linked
            return r;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = configured?.apply(this, arguments);
            // Restore by NAME when the workflow carries named values: positional widgets_values shift whenever a
            // widget is added/removed in a pack update, which would silently scramble saved settings.
            queueMicrotask(() => this.smPicker?.syncQuality());  // saved quality is set after the node is built
            const named = info?.widgets_values_named;
            if (named) {
                for (const w of this.widgets || []) {
                    if (w.name in named && w.name !== "pro_alpha_ui" && w.name !== "upload") w.value = named[w.name];
                }
                // Models saved from another machine (e.g. "D_drive\model.safetensors" vs "model.safetensors"):
                // select the listed file with the same name.
                for (const w of this.widgets || []) {
                    const vals = w.options?.values;
                    if (!Array.isArray(vals) || vals.includes(w.value) || typeof w.value !== "string") continue;
                    const base = w.value.split(/[\\/]/).pop();
                    const m = vals.find((v) => typeof v === "string" && v.split(/[\\/]/).pop() === base);
                    if (m) w.value = m;
                }
                // Workflows saved before "stride" became "overlap frames": the positional value of stride (40) lands in
                // the overlap slot. Convert: overlap = chunk - stride.
                const ov = this.widgets?.find((w) => w.name === "overlap");
                if (ov && !("overlap" in named)) {
                    const chunk = named.chunk ?? 49;
                    ov.value = "stride" in named ? Math.max(0, chunk - named.stride) : 9;
                }
            }
            if (this.smPicker) {
                this.smPicker.picks = this.smPicker.readPicks();
                this.smPicker.restore();
            }
            return r;
        };
        // Graph zoom changes the on-screen size without resizing the widget: re-fit the plate when it changes.
        const drawFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            drawFg?.apply(this, arguments);
            const p = this.smPicker;
            if (!p?.info) return;
            const k = p.screenScale();
            if (Math.abs(k - (p.lastScale || 0)) > 0.01) { p.lastScale = k; p.render(); }
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            executed?.apply(this, arguments);
            this.smPicker?.showResult(output);
        };
    },
});
