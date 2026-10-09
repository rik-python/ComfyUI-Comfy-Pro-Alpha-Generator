/**
 * Selective Matte picker + result viewer.
 *
 * Picker: scrub the plate, "Detect people" (SAM3 text "person" on this frame), click a person = target
 * (every other detected person is excluded automatically), Alt-click = cut a part (someone else's hand),
 * Shift-click = add a part (own hand SAM3 missed), Ctrl-click = remove the nearest cut/add.
 * Viewer: after a run, wipe between plate and matte comp, scrub, QC badge.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "SelectiveMatte";
const ADVANCED = ["ltx_model", "alpha_lora", "text_encoder", "video_vae", "audio_vae", "vitmatte", "chunk", "stride",
    "seed", "control_after_generate", "erode", "dilate", "near_px", "stabilize_radius", "sam3_threshold"];
const HIDDEN = ["picks", "render"];
const COLORS = ["#4aa3ff", "#b07cff", "#3fd0c9", "#e6d94a", "#ff6fb5", "#8bd450", "#ff9d5c", "#9aa4b2"];
const TARGET = "#ff8a1f", CUT = "#ff4d4d", ADD = "#3ddc84";

const css = `
.sm-root{display:flex;flex-direction:column;gap:6px;font:12px/1.3 system-ui,sans-serif;color:#ddd;width:100%;height:100%;box-sizing:border-box}
.sm-bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.sm-btn{background:#2b2f36;border:1px solid #444b55;color:#eee;border-radius:6px;padding:4px 10px;cursor:pointer;font:inherit}
.sm-btn:hover{background:#363b44}.sm-btn.primary{background:#ff8a1f;border-color:#ff8a1f;color:#111;font-weight:600}
.sm-btn.ghost{background:transparent}.sm-btn:disabled{opacity:.45;cursor:default}
.sm-view{position:relative;flex:1 1 auto;min-height:120px;background:#111;border-radius:8px;overflow:hidden}
.sm-view canvas{position:absolute;inset:0;width:100%;height:100%}
.sm-hint{position:absolute;left:8px;bottom:8px;right:8px;background:#000a;border-radius:6px;padding:4px 8px;pointer-events:none}
.sm-chip{display:inline-flex;gap:4px;align-items:center;background:#22262c;border-radius:10px;padding:2px 8px}
.sm-dot{width:8px;height:8px;border-radius:50%}
.sm-scrub{display:flex;gap:8px;align-items:center}.sm-scrub input{flex:1}
.sm-tabs{display:flex;gap:4px}.sm-tab{padding:3px 10px;border-radius:6px;cursor:pointer;background:#22262c}
.sm-tab.on{background:#ff8a1f;color:#111;font-weight:600}
.sm-qc{display:flex;gap:6px;flex-wrap:wrap}.sm-badge{border-radius:10px;padding:2px 8px;font-weight:600;color:#111}
.sm-g{background:#3ddc84}.sm-a{background:#ffc94d}.sm-r{background:#ff6b6b}
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
async function runHelper(prompt, reportId, key) {
    const res = await api.fetchApi("/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, client_id: api.clientId }),
    });
    const body = await res.json();
    if (!res.ok || body.error) throw new Error(body.error?.message || JSON.stringify(body.node_errors || body));
    const pid = body.prompt_id;
    return new Promise((resolve, reject) => {
        const onExec = (e) => {
            const d = e.detail;
            if (d.prompt_id === pid && d.node === reportId && d.output?.[key]) { done(); resolve(d.output[key][0]); }
        };
        const onErr = (e) => {
            if (e.detail?.prompt_id === pid) { done(); reject(new Error(e.detail.exception_message || "helper failed")); }
        };
        const done = () => { api.removeEventListener("executed", onExec); api.removeEventListener("execution_error", onErr); };
        api.addEventListener("executed", onExec);
        api.addEventListener("execution_error", onErr);
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

class Picker {
    constructor(node) {
        this.node = node;
        this.w = Object.fromEntries(node.widgets.map((w) => [w.name, w]));
        this.info = null;          // {frames, fps, width, height}
        this.frame = 0;
        this.img = null;           // current plate frame
        this.people = [];          // detections on picks.detect.frame: {id, box, area, mask, overlay}
        this.parts = {};           // token -> overlay
        this.result = null;        // {info, previews, view}
        this.tab = "pick";
        this.wipe = 0.5;
        this.picks = this.readPicks();
        this.build();
        this.loadVideo(false);
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

        const bar = el("div", "sm-bar");
        this.tabs = el("div", "sm-tabs");
        for (const [k, label] of [["pick", "Pick"], ["result", "Result"]]) {
            const t = el("div", "sm-tab", label);
            t.onclick = () => { this.tab = k; this.seek(this.frame); };
            t.dataset.k = k;
            this.tabs.appendChild(t);
        }
        this.detectBtn = el("button", "sm-btn primary", "Detect people");
        this.detectBtn.onclick = () => this.detect();
        const clear = el("button", "sm-btn ghost", "Clear picks");
        clear.onclick = () => { this.picks = { cuts: [], adds: [], others: [], detect: this.picks.detect }; this.writePicks(); this.render(); };
        this.previewBtn = el("button", "sm-btn", "Preview chunk");
        this.previewBtn.title = "Render the first 49 frames from the pick frame";
        this.previewBtn.onclick = () => this.queue(true);
        const run = el("button", "sm-btn", "Run full shot");
        run.onclick = () => this.queue(false);
        this.advBtn = el("button", "sm-btn ghost", "⚙ Advanced");
        this.advBtn.onclick = () => this.toggleAdvanced();
        bar.append(this.tabs, this.detectBtn, clear, this.previewBtn, run, this.advBtn);

        const view = el("div", "sm-view");
        this.canvas = el("canvas");
        this.hint = el("div", "sm-hint");
        view.append(this.canvas, this.hint);
        this.canvas.addEventListener("pointerdown", (e) => this.onPointer(e));
        this.canvas.addEventListener("pointermove", (e) => { if (e.buttons && this.tab === "result") this.onPointer(e); });
        this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
        new ResizeObserver(() => this.render()).observe(view);

        const scrub = el("div", "sm-scrub");
        this.slider = el("input");
        this.slider.type = "range"; this.slider.min = 0; this.slider.value = 0;
        this.slider.oninput = () => this.seek(+this.slider.value);
        this.frameLabel = el("span", null, "frame 1");
        scrub.append(this.slider, this.frameLabel);

        this.status = el("div", "sm-bar");
        this.qc = el("div", "sm-qc");
        root.append(bar, view, scrub, this.status, this.qc);

        // Fixed panel height (never derived from node.size: the node grows to fit its widgets -> feedback loop).
        this.widget = this.node.addDOMWidget("selective_matte_ui", "sm-ui", root, {
            serialize: false, hideOnZoom: false, getMinHeight: () => 560, getMaxHeight: () => 560,
        });

        HIDDEN.forEach((n) => hideWidget(this.w[n]));
        ADVANCED.forEach((n) => hideWidget(this.w[n]));
        const cb = this.w.video.callback;
        this.w.video.callback = (...a) => { const r = cb?.apply(this.w.video, a); this.loadVideo(true); return r; };
        this.fitNode();
        this.updateStatus();
    }

    /** The upload combo makes the frontend add its own video player widget; the picker already shows the plate. */
    dropVideoPreview() {
        const i = this.node.widgets.findIndex((w) => w.name === "video-preview");
        if (i < 0) return;
        const [w] = this.node.widgets.splice(i, 1);
        w.onRemove?.();
        w.element?.remove();
        this.fitNode();
    }

    fitNode() {
        const min = this.node.computeSize();
        this.node.setSize([Math.max(this.node.size[0], 640), min[1]]);
        app.graph?.setDirtyCanvas(true, true);
    }

    toggleAdvanced() {
        const show = !!this.w[ADVANCED[0]].sm_hidden;
        ADVANCED.forEach((n) => (show ? showWidget : hideWidget)(this.w[n]));
        this.advBtn.textContent = show ? "⚙ Hide advanced" : "⚙ Advanced";
        this.fitNode();
    }

    updateStatus() {
        const p = this.picks;
        const chip = (color, text) => {
            const c = el("span", "sm-chip");
            const d = el("span", "sm-dot"); d.style.background = color;
            c.append(d, document.createTextNode(text));
            return c;
        };
        this.status.replaceChildren(
            chip(TARGET, p.target ? `target #${p.target.id + 1} (frame ${p.target.frame + 1})` : "no target"),
            chip(COLORS[0], `excluded people ${p.others.length}`),
            chip(CUT, `cuts ${p.cuts.length}`),
            chip(ADD, `adds ${p.adds.length}`),
        );
        for (const t of this.tabs.children) t.classList.toggle("on", t.dataset.k === this.tab);
    }

    // ------------------------------------------------------------ data
    async loadVideo(reset) {
        [0, 300, 1500].forEach((t) => setTimeout(() => this.dropVideoPreview(), t));
        const name = this.w.video.value;
        if (!name) return;
        try {
            const res = await api.fetchApi(`/selective_matte/info?video=${encodeURIComponent(name)}`);
            if (!res.ok) throw new Error(await res.text());
            this.info = await res.json();
        } catch (e) { this.info = null; console.warn("[SelectiveMatte]", e); return; }
        if (reset) {
            this.picks = { cuts: [], adds: [], others: [] };
            this.people = []; this.parts = {}; this.result = null; this.tab = "pick";
            this.writePicks();
        }
        this.slider.max = this.info.frames - 1;
        await this.restoreDetections();
        this.seek(this.picks.target?.frame ?? this.picks.detect?.frame ?? 0);
    }

    async restoreDetections() {
        const d = this.picks.detect;
        this.people = [];
        if (!d) return;
        for (const saved of d.people) {
            const p = { ...saved };  // overlays stay out of the picks JSON (pixel arrays would bloat the workflow)
            try { p.overlay = tintMask(await loadImage(viewUrl(p.mask)), COLORS[p.id % COLORS.length]); }
            catch { p.overlay = null; }  // temp files are gone after a restart: fall back to boxes
            this.people.push(p);
        }
        this.framePeople = { [d.frame]: this.people };
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
        this.frameLabel.textContent = `frame ${this.frame + 1} / ${this.info.frames}`;
        if (this.tab === "pick") {
            const want = this.frame;
            try {
                const im = await loadImage(api.apiURL(`/selective_matte/frame?video=${encodeURIComponent(this.w.video.value)}&index=${want}`));
                if (want === this.frame) this.img = im;
            } catch (e) { console.warn("[SelectiveMatte] frame load failed", e); }
        }
        this.render();
    }

    /** SAM3 "person" detections on one frame (masks as overlays). Does not touch picks. */
    async detectOn(frame) {
        const token = Math.random().toString(36).slice(2, 10);
        const out = await runHelper({
            f: { class_type: "SelectiveMatte_Frame", inputs: { file: this.w.video.value, index: frame } },
            m: { class_type: "LoadSAM3Model", inputs: { precision: "auto" } },
            g: { class_type: "SAM3Grounding", inputs: { sam3_model_config: ["m", 0], image: ["f", 0], confidence_threshold: 0.3,
                                                       text_prompt: "person", max_detections: -1 } },
            r: { class_type: "SelectiveMatte_DetectReport", inputs: { masks: ["g", 0], frame, token } },
        }, "r", "selective_matte_detect");
        const people = out.people.sort((a, b) => a.box[0] - b.box[0]).map((p, i) => ({ ...p, id: i }));
        for (const p of people) p.overlay = tintMask(await loadImage(viewUrl(p.mask)), COLORS[p.id % COLORS.length]);
        return people;
    }

    async detect() {
        this.detectBtn.disabled = true;
        this.detectBtn.textContent = "Detecting…";
        try {
            const people = await this.detectOn(this.frame);
            this.framePeople = { [this.frame]: people };
            this.picks = { cuts: this.picks.cuts, adds: this.picks.adds, others: [],
                           detect: { frame: this.frame, people: people.map(({ overlay, ...p }) => p) } };
            this.people = people;
            this.writePicks();
            this.tab = "pick";
            this.render();
        } catch (e) {
            alert(`Selective Matte: detection failed\n${e.message}`);
        } finally {
            this.detectBtn.disabled = false;
            this.detectBtn.textContent = "Detect people";
        }
    }

    /** People on `frame` plus which of them is the target (nearest to the picked target box). */
    async peopleOn(frame) {
        this.framePeople = this.framePeople || {};
        if (frame === this.picks.detect?.frame && this.people.length) this.framePeople[frame] = this.people;
        if (!this.framePeople[frame]) this.framePeople[frame] = await this.detectOn(frame);
        const people = this.framePeople[frame];
        const t = this.picks.target;
        let target = null;
        if (t) {
            if (frame === t.frame) target = people.find((p) => p.id === t.id) || null;
            if (!target) {
                const cost = (p) => Math.hypot(p.box[0] - t.box[0], p.box[1] - t.box[1]) + Math.abs(p.box[3] - t.box[3]);
                target = [...people].sort((a, b) => cost(a) - cost(b))[0] || null;
            }
        }
        return { people, target };
    }

    async addPart(kind, x, y) {
        const token = Math.random().toString(36).slice(2, 10);
        const part = { frame: this.frame, x, y, token, points: [[x, y]], neg: [] };
        this.picks[kind].push(part);
        this.writePicks();
        this.render();
        try {
            // Auto negative points so one click gives a PART, never a whole person:
            //  add -> "not the neighbours" (centres of other people near the click)
            //  cut -> "not the subject" (centre of the target)
            const { people, target } = await this.peopleOn(part.frame);
            const near = (p) => {
                const [cx, cy, w, h] = p.box;
                return Math.abs(x - cx) < w / 2 + 0.05 && Math.abs(y - cy) < h / 2 + 0.05;
            };
            const negPeople = kind === "adds" ? people.filter((p) => p !== target && near(p)) : (target ? [target] : []);
            part.neg = negPeople.map((p) => interiorPoint(p.overlay, x, y)).filter(Boolean);

            const n = {
                f: { class_type: "SelectiveMatte_Frame", inputs: { file: this.w.video.value, index: part.frame } },
                m: { class_type: "LoadSAM3Model", inputs: { precision: "auto" } },
                p1: { class_type: "SAM3CreatePoint", inputs: { x, y, is_foreground: true } },
                c: { class_type: "SAM3CombinePoints", inputs: { point_1: ["p1", 0] } },
                s: { class_type: "SAM3Segmentation", inputs: { sam3_model_config: ["m", 0], image: ["f", 0], positive_points: ["c", 0] } },
                r: { class_type: "SelectiveMatte_PartReport", inputs: { mask: ["s", 0], token } },
            };
            if (part.neg.length) {
                const negInputs = {};
                part.neg.forEach(([nx, ny], i) => {
                    n[`n${i}`] = { class_type: "SAM3CreatePoint", inputs: { x: nx, y: ny, is_foreground: false } };
                    negInputs[`point_${i + 1}`] = [`n${i}`, 0];
                });
                n.cn = { class_type: "SAM3CombinePoints", inputs: negInputs };
                n.s.inputs.negative_points = ["cn", 0];
            }
            const out = await runHelper(n, "r", "selective_matte_part");
            part.mask = out.mask;
            const ov = tintMask(await loadImage(viewUrl(out.mask)), kind === "cuts" ? CUT : ADD);
            this.parts[token] = ov;
            // Safety: an add must not swallow another person, a cut must not swallow the subject.
            const victims = kind === "adds" ? people.filter((p) => p !== target) : (target ? [target] : []);
            const bad = victims.find((p) => coverage(ov, p.overlay) > 0.4);
            part.warning = bad ? (kind === "adds"
                ? `This add grabbed person #${bad.id + 1}. Ctrl-click it to remove, then click closer to the hand.`
                : "This cut grabbed the subject. Ctrl-click it to remove, then click right on the other person's hand.") : null;
            this.writePicks();
            this.render();
        } catch (e) {
            console.warn("[SelectiveMatte] part preview failed", e);
            part.warning = `Part preview failed: ${e.message}`;
            this.render();
        }
    }

    async queue(preview) {
        if (!this.picks.target) { alert("Selective Matte: click 'Detect people', then click the person to matte."); return; }
        const r = this.w.render;
        r.value = preview ? "preview chunk" : "full shot";
        try { await app.queuePrompt(0, 1); }
        finally { r.value = "full shot"; }
    }

    // ------------------------------------------------------------ interaction
    imageRect() {
        const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
        const iw = this.info?.width || 16, ih = this.info?.height || 9;
        const s = Math.min(cw / iw, ch / ih);
        const w = iw * s, h = ih * s;
        return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
    }

    onPointer(e) {
        const r = this.imageRect();
        const b = this.canvas.getBoundingClientRect();
        const x = (e.clientX - b.left - r.x) / r.w, y = (e.clientY - b.top - r.y) / r.h;
        if (this.tab === "result") { this.wipe = Math.max(0, Math.min(1, x)); this.render(); return; }
        if (x < 0 || x > 1 || y < 0 || y > 1 || e.button !== 0) return;
        e.preventDefault(); e.stopPropagation();
        if (e.ctrlKey || e.metaKey) return this.removeNearest(x, y);
        if (e.altKey) return this.addPart("cuts", x, y);
        if (e.shiftKey) return this.addPart("adds", x, y);
        this.pickPerson(x, y);
    }

    pickPerson(x, y) {
        const d = this.picks.detect;
        if (!d || d.frame !== this.frame) {
            alert(d ? `Detections are on frame ${d.frame + 1}. Scrub there, or Detect people on this frame.` : "Click 'Detect people' first.");
            return;
        }
        // Smallest person whose mask (or box) contains the click: front-most when people overlap.
        const hits = this.people.filter((p) => {
            if (p.overlay) {
                const o = p.overlay;
                return o.hit[Math.floor(y * o.h) * o.w + Math.floor(x * o.w)] === 1;
            }
            const [cx, cy, w, h] = p.box;
            return Math.abs(x - cx) < w / 2 && Math.abs(y - cy) < h / 2;
        }).sort((a, b) => a.area - b.area);
        if (!hits.length) return;
        const t = hits[0];
        this.picks.target = { id: t.id, frame: d.frame, box: t.box };
        // Only people near the target need painting out; far ones are dropped later by connectivity anyway,
        // and every excluded person costs one SAM3 track.
        const [tx, ty, tw, th] = t.box, gx = tw * 0.65, gy = th * 0.65;
        const near = (p) => Math.abs(p.box[0] - tx) < gx + p.box[2] / 2 && Math.abs(p.box[1] - ty) < gy + p.box[3] / 2;
        this.picks.others = this.people.filter((p) => p.id !== t.id && near(p)).map((p) => ({ id: p.id, frame: d.frame, box: p.box }));
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
        this.previewBtn.disabled = !this.picks.target;
        if (this.tab === "result") return this.renderResult(g);
        const r = this.imageRect();
        if (this.img) g.drawImage(this.img, r.x, r.y, r.w, r.h);

        const d = this.picks.detect;
        const onDetect = d && d.frame === this.frame;
        if (onDetect) {
            for (const p of this.people) {
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
                g.fillStyle = color;
                g.beginPath(); g.arc(bx + 12, by + 12, 10, 0, Math.PI * 2); g.fill();
                g.fillStyle = "#111"; g.font = "bold 12px system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
                g.fillText(String(p.id + 1), bx + 12, by + 12);
            }
        }
        for (const [kind, color, sym] of [["cuts", CUT, "×"], ["adds", ADD, "+"]]) {
            for (const p of this.picks[kind]) {
                if (p.frame !== this.frame) continue;
                const ov = this.parts[p.token];
                if (ov) { g.globalAlpha = 0.5; g.drawImage(ov.canvas, r.x, r.y, r.w, r.h); g.globalAlpha = 1; }
                const px = r.x + p.x * r.w, py = r.y + p.y * r.h;
                g.fillStyle = color; g.beginPath(); g.arc(px, py, 9, 0, Math.PI * 2); g.fill();
                g.fillStyle = "#111"; g.font = "bold 14px system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
                g.fillText(sym, px, py + 1);
            }
        }
        const warn = [...this.picks.cuts, ...this.picks.adds].find((p) => p.warning);
        this.hint.style.background = warn ? "#7a1f1fdd" : "";
        this.hint.textContent = warn ? `⚠ ${warn.warning}`
            : !this.info ? "Upload or choose a video above."
            : !d ? "Scrub to a frame where the person is fully visible, then click Detect people."
            : !onDetect ? `Detections live on frame ${d.frame + 1}. Cuts/adds can go on any frame (place them where the part first appears).`
            : !this.picks.target ? "Click the person to matte. Every other detected person is excluded."
            : "Alt-click: cut a part (someone else's hand).  Shift-click: add a part (own hand).  Ctrl-click: remove.";
    }

    renderResult(g) {
        const res = this.result;
        const r = this.imageRect();
        if (!res || !res.previews.length) { this.hint.textContent = "Run Preview chunk or the full shot to see the result here."; return; }
        const items = res.previews;
        const first = res.info.first_frame - 1;
        // nearest preview to the scrubber frame
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
        this.hint.textContent = `${q.mode} · frame ${q.first_frame + it.frame} · drag to wipe plate | ${key}` +
            (q.out_dir ? ` · saved: ${q.out_dir}` : "");
    }

    showResult(output) {
        const info = output?.selective_matte?.[0];
        if (!info) return;
        this.result = { info, previews: output.previews || [], view: this.result?.view || "comp" };
        this.tab = "result";
        const badge = (label, value, cls) => el("span", `sm-badge ${cls}`, `${label} ${value}`);
        const kids = [];
        if (info.flicker_x != null) {
            const f = info.flicker_x;
            kids.push(badge("flicker", `${f.toFixed(2)}× LTX`, f <= 1.3 ? "sm-g" : f <= 2 ? "sm-a" : "sm-r"));
        }
        if (info.hair_pct != null) {
            const h = info.hair_pct;
            kids.push(badge("hair kept", `${h.toFixed(1)}%`, h >= 95 ? "sm-g" : h >= 85 ? "sm-a" : "sm-r"));
        }
        const tog = el("button", "sm-btn ghost", "comp / alpha");
        tog.onclick = () => { this.result.view = this.result.view === "alpha" ? "comp" : "alpha"; this.render(); };
        kids.push(tog);
        this.qc.replaceChildren(...kids);
        this.seek(info.first_frame - 1);
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
    name: "SelectiveMatte.Picker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created?.apply(this, arguments);
            this.smPicker = new Picker(this);
            return r;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = configured?.apply(this, arguments);
            // Restore by NAME when the workflow carries named values: positional widgets_values shift whenever a
            // widget is added/removed in a pack update, which would silently scramble saved settings.
            const named = info?.widgets_values_named;
            if (named) {
                for (const w of this.widgets || []) {
                    if (w.name in named && w.name !== "selective_matte_ui" && w.name !== "upload") w.value = named[w.name];
                }
            }
            if (this.smPicker) { this.smPicker.picks = this.smPicker.readPicks(); this.smPicker.loadVideo(false); }
            return r;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            executed?.apply(this, arguments);
            this.smPicker?.showResult(output);
        };
    },
});
