from __future__ import annotations

import dataclasses
import http.server
import json
import pathlib
from typing import Any

from system1_eval.runner import run_benchmark
from system1_eval.schema import load_dataset_dir

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>System1-Eval (DecisionArena) — Canlı Liderlik Tablosu & Kalibrasyon</title>
  <style>
    :root {
      --bg: #0f1117;
      --panel: #181b24;
      --panel-alt: #202431;
      --border: #2c3244;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #3b82f6;
      --green: #10b981;
      --amber: #f59e0b;
      --red: #ef4444;
      --purple: #a855f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 1.5rem 2rem;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 1rem;
      padding-bottom: 1.25rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1.5rem;
    }
    h1 { font-size: 1.55rem; font-weight: 700; display: flex; align-items: center; gap: 0.6rem; }
    .badge {
      font-size: 0.75rem;
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.4);
      padding: 0.2rem 0.6rem;
      border-radius: 999px;
      font-weight: 600;
    }
    .subtitle { color: var(--muted); font-size: 0.9rem; margin-top: 0.3rem; }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: center;
      margin-bottom: 1.25rem;
      background: var(--panel);
      padding: 0.85rem 1rem;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .tab-btn {
      background: var(--panel-alt);
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 0.45rem 0.85rem;
      border-radius: 8px;
      font-size: 0.82rem;
      cursor: pointer;
      font-weight: 500;
      transition: all 0.15s;
    }
    .tab-btn.active, .tab-btn:hover {
      background: rgba(59, 130, 246, 0.2);
      color: #fff;
      border-color: var(--accent);
    }
    select, input[type="number"] {
      background: var(--panel-alt);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.4rem 0.65rem;
      border-radius: 6px;
      font-size: 0.82rem;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.45fr 1fr;
      gap: 1.25rem;
      margin-bottom: 1.5rem;
    }
    @media (max-width: 1024px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.2rem;
    }
    .card h2 {
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 0.3rem;
    }
    .card p.desc {
      font-size: 0.8rem;
      color: var(--muted);
      margin-bottom: 1rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
    }
    th, td {
      padding: 0.65rem 0.5rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th { color: var(--muted); font-weight: 600; font-size: 0.76rem; text-transform: uppercase; }
    tr:hover { background: rgba(255,255,255,0.02); }
    .pill-s1 {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.35);
      padding: 0.15rem 0.45rem;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 600;
    }
    .pill-llm {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.35);
      padding: 0.15rem 0.45rem;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 600;
    }
    .good { color: var(--green); font-weight: 600; }
    .warn { color: var(--amber); font-weight: 600; }
    .bad { color: var(--red); font-weight: 600; }
    .scenarios-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
      gap: 0.9rem;
      margin-top: 0.75rem;
    }
    .scenario-box {
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.9rem;
      font-size: 0.8rem;
    }
    .scenario-box pre {
      background: #0b0d12;
      padding: 0.55rem;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 0.74rem;
      margin-top: 0.45rem;
      color: #cbd5e1;
    }
    .btn-run {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.84rem;
    }
    .btn-run:hover { opacity: 0.92; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>⚡ System1-Eval (DecisionArena) <span class="badge">/v1/systemone Benchmark</span></h1>
      <p class="subtitle">Clef, Kev, Jev, Tev1 ve açık ağırlıklı modellerin Doğruluk, Olasılık Kalibrasyonu (ECE / Brier) ve Gecikme (ms) kıyaslama platformu</p>
    </div>
    <div>
      <button class="btn-run" onclick="triggerReEvaluation()">🔄 Değerlendirmeyi Yeniden Koştur</button>
    </div>
  </header>

  <div class="controls">
    <span style="font-size:0.8rem; color:var(--muted); font-weight:600;">Senaryo Havuzu:</span>
    <button class="tab-btn active" data-track="overall" onclick="setTrack('overall', this)">Tümü (30 Senaryo)</button>
    <button class="tab-btn" data-track="agent_routing" onclick="setTrack('agent_routing', this)">🧭 Ajan Yönlendirme (10)</button>
    <button class="tab-btn" data-track="security_guardrail" onclick="setTrack('security_guardrail', this)">🛡️ Güvenlik / Guardrail (10)</button>
    <button class="tab-btn" data-track="structured_triage" onclick="setTrack('structured_triage', this)">📄 Veri Yapılandırma (10)</button>

    <div style="margin-left:auto; display:flex; gap:0.6rem; align-items:center; flex-wrap:wrap;">
      <label style="font-size:0.78rem; color:var(--muted);">Model Tipi:
        <select id="typeFilter" onchange="renderAll()">
          <option value="all">Tümü (System-1 + LLM)</option>
          <option value="system1">Sadece System-1 (Clef/Kev/Jev/Tev1)</option>
          <option value="llm_structured">Sadece LLM Structured JSON</option>
        </select>
      </label>
      <label style="font-size:0.78rem; color:var(--muted);">Sıralama:
        <select id="sortBy" onchange="renderAll()">
          <option value="arena">Arena Skoru (Doğruluk + Kalibrasyon)</option>
          <option value="accuracy">Karar Doğruluğu (Acc % ↑)</option>
          <option value="ece">Olasılık Kalibrasyonu (ECE ↓)</option>
          <option value="latency">Gecikme Süresi (p50 ms ↓)</option>
          <option value="cost">1K Karar Maliyeti ($ ↓)</option>
        </select>
      </label>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>🏆 Canlı Dinamik Liderlik Tablosu (Leaderboard)</h2>
      <p class="desc">Grafikte karşılaştırmak istediğin modelleri tablodaki onay kutularından seçebilirsin.</p>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Grafik</th>
              <th>#</th>
              <th>Model</th>
              <th>Mimari</th>
              <th>Acc (%) ↑</th>
              <th>ECE ↓ (Kalibrasyon)</th>
              <th>Brier ↓</th>
              <th>p50 / p95 Hız ↓</th>
              <th>Çıktı Token</th>
              <th>Maliyet / 1K</th>
            </tr>
          </thead>
          <tbody id="leaderboardBody"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h2>📈 Reliability Diagram (Olasılık Kalibrasyon Eğrisi)</h2>
      <p class="desc">Model <strong>"%90 eminim"</strong> dediğinde gerçekten <strong>%90</strong> oranında haklı mı? Kesikli köşegen (y = x) kusursuz kalibrasyonu temsil eder.</p>
      <svg id="calibrationSvg" viewBox="0 0 380 280" style="width:100%; height:290px; background:#0b0d12; border-radius:8px; border:1px solid var(--border);"></svg>
      <div id="legendBox" style="display:flex; flex-wrap:wrap; gap:0.75rem; margin-top:0.75rem; font-size:0.78rem;"></div>
    </div>
  </div>

  <div class="card">
    <h2>📚 Topluluk Katkılı Senaryo Havuzu (Crowdsourced Dataset Explorer)</h2>
    <p class="desc">Seçili kategorideki gerçek dünya test senaryoları, <code>/v1/systemone</code> soru şemaları ve kesin doğruluk (ground-truth) kriterleri:</p>
    <div id="scenariosContainer" class="scenarios-grid"></div>
  </div>

  <script>
    let leaderboardData = null;
    let scenariosData = [];
    let currentTrack = 'overall';
    let selectedModels = new Set(['kev-4b-v1', 'clef-flash', 'jev-latest', 'llama-3.1-8b-vllm']);
    const COLORS = ['#10b981', '#3b82f6', '#a855f7', '#ef4444', '#f59e0b', '#06b6d4', '#ec4899', '#84cc16'];

    async function loadInitialData() {
      const [lbRes, scRes] = await Promise.all([
        fetch('/api/leaderboard'),
        fetch('/api/scenarios')
      ]);
      leaderboardData = await lbRes.json();
      const scJson = await scRes.json();
      scenariosData = scJson.scenarios || [];
      renderAll();
    }

    function setTrack(track, btn) {
      currentTrack = track;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderAll();
    }

    function toggleModelPlot(modelId, checkbox) {
      if (checkbox.checked) selectedModels.add(modelId);
      else selectedModels.delete(modelId);
      renderCalibrationChart();
    }

    function getMetricsForTrack(model) {
      if (currentTrack === 'overall') return model.overall;
      return (model.tracks && model.tracks[currentTrack]) ? model.tracks[currentTrack] : model.overall;
    }

    function renderAll() {
      if (!leaderboardData) return;
      const typeFilter = document.getElementById('typeFilter').value;
      const sortBy = document.getElementById('sortBy').value;

      let models = leaderboardData.models.filter(m => {
        if (typeFilter === 'all') return true;
        return m.model_type === typeFilter;
      });

      models.sort((a, b) => {
        const ma = getMetricsForTrack(a);
        const mb = getMetricsForTrack(b);
        if (sortBy === 'accuracy') return mb.accuracy - ma.accuracy;
        if (sortBy === 'ece') return ma.ece - mb.ece;
        if (sortBy === 'latency') return ma.p50_ms - mb.p50_ms;
        if (sortBy === 'cost') return ma.cost_per_1k_usd - mb.cost_per_1k_usd;
        const scoreA = (ma.accuracy * 100) - (ma.ece * 50);
        const scoreB = (mb.accuracy * 100) - (mb.ece * 50);
        return scoreB - scoreA;
      });

      const tbody = document.getElementById('leaderboardBody');
      tbody.innerHTML = models.map((m, idx) => {
        const met = getMetricsForTrack(m);
        const accPct = (met.accuracy * 100).toFixed(1) + '%';
        const eceClass = met.ece < 0.05 ? 'good' : (met.ece < 0.10 ? 'warn' : 'bad');
        const pill = m.model_type === 'system1'
          ? '<span class="pill-s1">System-1</span>'
          : '<span class="pill-llm">LLM JSON</span>';
        const checked = selectedModels.has(m.model_id) ? 'checked' : '';
        return `
          <tr>
            <td><input type="checkbox" ${checked} onchange="toggleModelPlot('${m.model_id}', this)" /></td>
            <td><strong>${idx + 1}</strong></td>
            <td>
              <div style="font-weight:600;">${m.display_name}</div>
              <div style="font-size:0.72rem; color:var(--muted);">${m.provider} (${m.parameters_b}B)</div>
            </td>
            <td>${pill}</td>
            <td style="font-weight:600;">${accPct}</td>
            <td class="${eceClass}">${met.ece.toFixed(4)}</td>
            <td>${met.brier_score.toFixed(4)}</td>
            <td>${met.p50_ms.toFixed(1)} / ${met.p95_ms.toFixed(1)} ms</td>
            <td class="${met.avg_output_tokens === 0 ? 'good' : 'warn'}">${met.avg_output_tokens} tok</td>
            <td>$${met.cost_per_1k_usd.toFixed(4)}</td>
          </tr>
        `;
      }).join('');

      renderCalibrationChart();
      renderScenarios();
    }

    function renderCalibrationChart() {
      const svg = document.getElementById('calibrationSvg');
      const legendBox = document.getElementById('legendBox');
      const padL = 45, padR = 20, padT = 20, padB = 38;
      const W = 380 - padL - padR;
      const H = 280 - padT - padB;

      let parts = [];
      for (let i = 0; i <= 5; i++) {
        const v = i / 5;
        const x = padL + v * W;
        const y = padT + (1 - v) * H;
        parts.push(`<line x1="${padL}" y1="${y}" x2="${padL + W}" y2="${y}" stroke="#1e2433" stroke-width="1"/>`);
        parts.push(`<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + H}" stroke="#1e2433" stroke-width="1"/>`);
        parts.push(`<text x="${padL - 8}" y="${y + 3}" fill="#94a3b8" font-size="9" text-anchor="end">${(v*100).toFixed(0)}%</text>`);
        parts.push(`<text x="${x}" y="${padT + H + 14}" fill="#94a3b8" font-size="9" text-anchor="middle">${(v*100).toFixed(0)}%</text>`);
      }

      // Ideal diagonal y = x
      parts.push(`<line x1="${padL}" y1="${padT + H}" x2="${padL + W}" y2="${padT}" stroke="#64748b" stroke-dasharray="5,4" stroke-width="1.5"/>`);
      parts.push(`<text x="${padL + W / 2}" y="272" fill="#94a3b8" font-size="10" text-anchor="middle">Beyan Edilen Olasılık / Özgüven (Confidence)</text>`);

      let legendHtml = [];
      leaderboardData.models.forEach((m, idx) => {
        if (!selectedModels.has(m.model_id)) return;
        const color = COLORS[idx % COLORS.length];
        const met = getMetricsForTrack(m);
        const nonEmpty = (met.calibration_bins || []).filter(b => b.count > 0);
        if (nonEmpty.length === 0) return;

        const pts = nonEmpty.map(b => {
          const x = padL + b.avg_conf * W;
          const y = padT + (1 - b.avg_acc) * H;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(' ');

        parts.push(`<polyline fill="none" stroke="${color}" stroke-width="2.5" points="${pts}"/>`);
        nonEmpty.forEach(b => {
          const x = padL + b.avg_conf * W;
          const y = padT + (1 - b.avg_acc) * H;
          parts.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" fill="${color}"/>`);
        });

        legendHtml.push(`<span style="display:flex;align-items:center;gap:0.35rem;"><span style="width:10px;height:10px;border-radius:50%;background:${color};display:inline-block;"></span><strong>${m.model_id}</strong> (ECE: ${met.ece.toFixed(3)})</span>`);
      });

      svg.innerHTML = parts.join('');
      legendBox.innerHTML = legendHtml.join('');
    }

    function renderScenarios() {
      const container = document.getElementById('scenariosContainer');
      const filtered = scenariosData.filter(s => currentTrack === 'overall' || s.track === currentTrack);
      container.innerHTML = filtered.slice(0, 12).map(s => `
        <div class="scenario-box">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong>${s.id} — ${s.title}</strong>
            <span class="badge">${s.question.type}</span>
          </div>
          <div style="color:var(--muted); margin-top:0.25rem;">Kategori: <code>${s.track}</code></div>
          <pre>${JSON.stringify({state: s.state, question: {type: s.question.type, instructions: s.question.instructions}, ground_truth: s.ground_truth}, null, 2)}</pre>
        </div>
      `).join('');
    }

    async function triggerReEvaluation() {
      const res = await fetch('/api/evaluate', { method: 'POST' });
      leaderboardData = await res.json();
      renderAll();
    }

    loadInitialData();
  </script>
</body>
</html>
"""


def create_app_server(
    host: str,
    port: int,
    dataset_dir: pathlib.Path | str,
    results_path: pathlib.Path | str,
) -> http.server.ThreadingHTTPServer:
    ds_path = pathlib.Path(dataset_dir)
    res_path = pathlib.Path(results_path)

    if not res_path.exists():
        run_benchmark(dataset_dir=ds_path, output_path=res_path)

    class ArenaHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _send_json(self, data: Any, status: int = 200) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/api/leaderboard"):
                if not res_path.exists():
                    run_benchmark(dataset_dir=ds_path, output_path=res_path)
                data = json.loads(res_path.read_text(encoding="utf-8"))
                self._send_json(data)
                return

            if self.path.startswith("/api/scenarios"):
                scenarios = load_dataset_dir(ds_path)
                self._send_json(
                    {
                        "count": len(scenarios),
                        "scenarios": [dataclasses.asdict(s) for s in scenarios],
                    }
                )
                return

            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path.startswith("/api/evaluate"):
                updated = run_benchmark(dataset_dir=ds_path, output_path=res_path)
                self._send_json(updated)
                return
            self._send_json({"error": "Not found"}, status=404)

    http.server.ThreadingHTTPServer.allow_reuse_address = True
    return http.server.ThreadingHTTPServer((host, port), ArenaHandler)
