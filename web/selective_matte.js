/**
 * Selective Matte picker + result viewer.
 *
 * Picker: scrub the plate, type what to find ("person", "dog, bike"...), Detect (SAM3 text detection on this
 * frame), click a subject = target (other detections near it are excluded automatically),
 * Alt-click = cut a part (someone else's hand),
 * Shift-click = add a part (own hand SAM3 missed), Ctrl-click = remove the nearest cut/add.
 * Viewer: after a run, wipe between plate and matte comp, scrub, QC badge.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "SelectiveMatte";
const ADVANCED = ["chunk", "overlap", "fast_attention",
    "seed", "control_after_generate", "erode", "dilate", "near_px", "stabilize_radius", "sam3_threshold"];
const HIDDEN = ["picks", "render"];
const COLORS = ["#4aa3ff", "#b07cff", "#3fd0c9", "#e6d94a", "#ff6fb5", "#8bd450", "#ff9d5c", "#9aa4b2"];
const TARGET = "#ff8a1f", CUT = "#ff4d4d", ADD = "#3ddc84";
const MAX_OTHERS = 8;  // excluded detections (one SAM3 track each)

const css = `
.sm-root{display:flex;flex-direction:column;gap:6px;font:12px/1.3 system-ui,sans-serif;color:#ddd;width:100%;height:100%;box-sizing:border-box}
.sm-bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.sm-btn{background:#2b2f36;border:1px solid #444b55;color:#eee;border-radius:6px;padding:4px 10px;cursor:pointer;font:inherit}
.sm-btn:hover{background:#363b44}.sm-btn.primary{background:#ff8a1f;border-color:#ff8a1f;color:#111;font-weight:600}
.sm-btn.ghost{background:transparent}
.sm-input{background:#1b1e23;border:1px solid #444b55;color:#eee;border-radius:6px;padding:4px 8px;font:inherit;width:150px}.sm-btn:disabled{opacity:.45;cursor:default}
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
        this.tab = "pick";
        this.wipe = 0.5;
        this.picks = this.readPicks();
        this.build();
        this.restore();
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
        this.loadBtn = el("button", "sm-btn", "Load plate");
        this.loadBtn.title = "Run the nodes feeding 'images' and show the plate here (again after changing crop/resize)";
        this.loadBtn.onclick = () => this.loadPlate();
        this.promptBox = el("input", "sm-input");
        this.promptBox.placeholder = "person, dog, bike…";
        this.promptBox.title = "What to detect. Comma-separate several (e.g. 'dog, person' to also exclude people)";
        this.promptBox.value = this.picks.prompt || this.picks.detect?.prompt || "person";
        // Keys typed here must not reach the graph canvas (Delete, Ctrl+C, space...).
        for (const ev of ["keydown", "keyup", "keypress"]) this.promptBox.addEventListener(ev, (e) => e.stopPropagation());
        this.promptBox.addEventListener("keydown", (e) => { if (e.key === "Enter") this.detect(); });
        this.promptBox.addEventListener("change", () => { this.picks.prompt = this.promptText(); this.writePicks(); });
        this.detectBtn = el("button", "sm-btn primary", "Detect");
        this.detectBtn.onclick = () => this.detect();
        const clear = el("button", "sm-btn ghost", "Clear picks");
        clear.onclick = () => {
            this.picks = { cuts: [], adds: [], others: [], detect: this.picks.detect, prompt: this.picks.prompt };
            this.writePicks(); this.render();
        };
        this.previewBtn = el("button", "sm-btn", "Preview chunk");
        this.previewBtn.title = "Render the first 49 frames from the pick frame";
        this.previewBtn.onclick = () => this.queue(true);
        const run = el("button", "sm-btn", "Run full shot");
        run.onclick = () => this.queue(false);
        this.advBtn = el("button", "sm-btn ghost", "⚙ Advanced");
        this.advBtn.onclick = () => this.toggleAdvanced();
        bar.append(this.tabs, this.loadBtn, this.promptBox, this.detectBtn, clear, this.previewBtn, run, this.advBtn);

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

        // Min height only, no max: the frontend gives the panel all spare node height, so it grows/shrinks when the
        // node is resized. (Never derive the height from node.size ourselves: the node grows to fit -> feedback loop.)
        this.widget = this.node.addDOMWidget("selective_matte_ui", "sm-ui", root, {
            serialize: false, hideOnZoom: false, getMinHeight: () => 420,
        });

        HIDDEN.forEach((n) => hideWidget(this.w[n]));
        ADVANCED.forEach((n) => hideWidget(this.w[n]));
        this.fitNode();
        this.updateStatus();
    }

    /** Plate frame number of index i (first_frame, e.g. 1001 for VFX plates). */
    num(i) { return (this.info?.first_frame ?? this.w.first_frame?.value ?? 1) + i; }

    /** API prompt of everything feeding our `images` (and first_frame) input, plus that link. */
    async upstream() {
        const { output } = await app.graphToPrompt();
        const me = output[String(this.node.id)];
        if (!me) throw new Error("The Selective Matte node is muted or bypassed.");
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
        this.loadBtn.textContent = "Loading…";
        try {
            const info = await this.helper((images, firstFrame) => ({
                pp: { class_type: "SelectiveMatte_PlateProxy", inputs: { images, width: 960, first_frame: firstFrame } },
            }), "pp", "selective_matte_plate");
            const changed = this.picks.view && this.picks.view.fingerprint !== info.fingerprint;
            this.info = info;
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
            alert(`Selective Matte: could not load the plate\n${e.message}`);
        } finally {
            this.loadBtn.disabled = false;
            this.loadBtn.textContent = "Load plate";
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
        const kids = [];
        if (this.info) {  // LTX alpha-gen is tuned for 1920x1088; a 1920x1080 plate is padded to it automatically
            const { width: w, height: h } = this.info;
            const best = w === 1920 && (h === 1080 || h === 1088);
            const c = chip(best ? ADD : "#ffc94d", best ? `plate ${w}×${h}` : `plate ${w}×${h} (best: 1920×1088)`);
            c.title = best ? "Best size for LTX alpha-gen."
                : "LTX alpha-gen is tuned for 1920x1088 (1080p plates are padded automatically). Other sizes work, "
                  + "but edges can be softer. Resize upstream for best quality; for 4K, matte a 1080p proxy.";
            kids.push(c);
        }
        this.status.replaceChildren(
            ...kids,
            chip(TARGET, !p.target ? "no target"
                : p.target.box ? `target #${p.target.id + 1} (frame ${this.num(p.target.frame)})`
                : `target: clicked object (frame ${this.num(p.target.frame)})`),
            chip(COLORS[0], `excluded ${p.others.length}`),
            chip(CUT, `cuts ${p.cuts.length}`),
            chip(ADD, `adds ${p.adds.length}`),
        );
        for (const t of this.tabs.children) t.classList.toggle("on", t.dataset.k === this.tab);
    }

    // ------------------------------------------------------------ data
    async restore() {
        this.info = this.picks.view || null;
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
                f: { class_type: "SelectiveMatte_Frame", inputs: { images, index: frame } },
                m: { class_type: "LoadSAM3Model", inputs: { precision: "auto" } },
                g: { class_type: "SAM3Grounding", inputs: { sam3_model_config: ["m", 0], image: ["f", 0], confidence_threshold: 0.3,
                                                           text_prompt: term, max_detections: -1 } },
                r: { class_type: "SelectiveMatte_DetectReport", inputs: { masks: ["g", 0], frame, token } },
            }), "r", "selective_matte_detect");
            found.push(...(out.subjects || out.people || []).map((p) => ({ ...p, label: term })));
        }
        const subjects = found.sort((a, b) => a.box[0] - b.box[0]).map((p, i) => ({ ...p, id: i }));
        for (const p of subjects) p.overlay = tintMask(await loadImage(viewUrl(p.mask)), COLORS[p.id % COLORS.length]);
        return subjects;
    }

    /** quiet: auto-detect on a click (no alert when nothing is found). */
    async detect(quiet = false) {
        this.detectBtn.disabled = true;
        this.detectBtn.textContent = "Detecting…";
        try {
            if (!this.info) await this.loadPlate();
            if (!this.info) return;
            const prompt = this.promptText();
            this.promptBox.value = prompt;
            const subjects = await this.detectOn(this.frame, prompt);
            if (!subjects.length) {
                if (!quiet) alert(`Selective Matte: no "${prompt}" found on this frame. Try another word or frame.`);
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
            alert(`Selective Matte: detection failed\n${e.message}`);
        } finally {
            this.detectBtn.disabled = false;
            this.detectBtn.textContent = "Detect";
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
        const n = {
            f: { class_type: "SelectiveMatte_Frame", inputs: { images: null, index: frame } },
            m: { class_type: "LoadSAM3Model", inputs: { precision: "auto" } },
            p1: { class_type: "SAM3CreatePoint", inputs: { x, y, is_foreground: true } },
            c: { class_type: "SAM3CombinePoints", inputs: { point_1: ["p1", 0] } },
            s: { class_type: "SAM3Segmentation", inputs: { sam3_model_config: ["m", 0], image: ["f", 0], positive_points: ["c", 0] } },
            r: { class_type: "SelectiveMatte_PartReport", inputs: { mask: ["s", 0], token } },
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
        }, "r", "selective_matte_part");
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
            console.warn("[SelectiveMatte] part preview failed", e);
            part.warning = `Part preview failed: ${e.message}`;
            this.render();
        }
    }

    async queue(preview) {
        if (!this.picks.target) { alert("Selective Matte: click the subject to matte (anything in the plate, or a detection after Detect)."); return; }
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
            console.warn("[SelectiveMatte] target preview failed", e);
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
        this.previewBtn.disabled = !this.picks.target;
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
                g.font = "bold 12px system-ui"; g.textAlign = "left"; g.textBaseline = "middle";
                const tw = g.measureText(tag).width + 12;
                g.fillStyle = color;
                g.beginPath(); g.roundRect(bx + 2, by + 2, tw, 20, 10); g.fill();
                g.fillStyle = "#111";
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
                g.fillStyle = "#111"; g.font = "bold 14px system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
                g.fillText(sym, px, py + 1);
            }
        }
        if (this.waiting) {
            this.hint.style.background = "#5a4a12dd";
            this.hint.textContent = "⏳ Waiting for the running render to finish (ComfyUI runs one job at a time). "
                + "Cancel it in the queue panel to go now.";
            return;
        }
        const warn = [...this.picks.cuts, ...this.picks.adds].find((p) => p.warning);
        this.hint.style.background = warn || this.picks.stale ? "#7a1f1fdd" : "";
        this.hint.textContent = warn ? `⚠ ${warn.warning}`
            : this.picks.stale ? "⚠ The plate changed (crop/resize/new clip) after picking: Detect again."
            : !this.info ? "Connect the plate to 'images', then click Load plate."
            : !this.picks.target && !d ? "Click anything to matte it, or type what to find (person, dog, car…) and click Detect."
            : !this.picks.target ? `Detections live on frame ${this.num(d.frame)}. Click a subject there, or click anything on this frame.`
            : !onDetect && this.picks.target.frame !== this.frame ? "Cuts/adds can go on any frame (place them where the part first appears)."
            : !this.picks.target ? "Click a detection, or anything else, to matte it. Other detections are excluded."
            : "Shift-click: add more (the glass, the table).  Alt-click: cut a part.  Ctrl-click: remove.";
    }

    renderResult(g) {
        const res = this.result;
        const r = this.imageRect();
        if (!res || !res.previews.length) { this.hint.textContent = "Run Preview chunk or the full shot to see the result here."; return; }
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
        this.hint.style.background = inside ? "" : "#5a4a12dd";
        this.hint.textContent = inside
            ? `${range} · ${q.mode} · frame ${q.first_frame + it.frame} · drag to wipe plate | ${key}` +
              (q.out_dir ? ` · saved: ${q.out_dir}` : "")
            : `Frame ${this.num(this.frame)} is outside this render (${range}). Showing the nearest rendered frame.` +
              (kind === "preview" ? " Run full shot to render the rest." : "");
    }

    showResult(output) {
        const info = output?.selective_matte?.[0];
        if (!info) return;
        this.result = { info, previews: output.previews || [], view: this.result?.view || "comp" };
        this.tab = "result";
        const badge = (label, value, cls) => el("span", `sm-badge ${cls}`, `${label} ${value}`);
        const kids = [];
        // Flicker and soft edge are measured against LTX alone: meaningless when LTX barely matted the subject.
        const vsLtx = info.ltx_cover == null || info.ltx_cover >= 60;
        if (!vsLtx) {
            const b = badge("flicker / soft edge", "n/a", "");
            b.style.background = "#9aa4b2";
            b.title = "Measured against LTX, which did not matte this subject: mostly a ViTMatte result. Judge by eye.";
            kids.push(b);
        } else if (info.flicker_x != null) {
            const f = info.flicker_x;
            kids.push(badge("flicker", `${f.toFixed(2)}× LTX`, f <= 1.3 ? "sm-g" : f <= 2 ? "sm-a" : "sm-r"));
        }
        if (vsLtx && info.hair_pct != null) {
            const h = Math.min(info.hair_pct, 100);  // the isolated pass can keep MORE soft edge than LTX alone
            kids.push(badge("soft edge kept", `${h.toFixed(1)}%`, h >= 95 ? "sm-g" : h >= 85 ? "sm-a" : "sm-r"));
        }
        if (info.ltx_cover != null) {
            const c = info.ltx_cover;
            const b = badge("LTX cover", `${c.toFixed(0)}%`, c >= 90 ? "sm-g" : c >= 60 ? "sm-a" : "sm-r");
            b.title = "How much of the subject LTX alpha-gen matted. Low: LTX did not see it as the main subject, "
                    + "so those frames use the ViTMatte edge instead.";
            kids.push(b);
        }
        const tog = el("button", "sm-btn ghost", "comp / alpha");
        tog.onclick = () => { this.result.view = this.result.view === "alpha" ? "comp" : "alpha"; this.render(); };
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
