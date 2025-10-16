from flask import Flask, request, render_template_string, Response, jsonify, send_file
from flask_cors import CORS
import subprocess
import sys
import threading
import time
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

process = None
log_buffer = []
lock = threading.Lock()

DB_PATH = 'lost_items.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lost_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persistent_id INTEGER,
                class_name TEXT NOT NULL,
                location TEXT,
                first_seen_ts TEXT NOT NULL,
                last_seen_ts TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_status ON lost_events(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_persistent ON lost_events(persistent_id)")
        conn.commit()
    finally:
        conn.close()

def clear_old_events():
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM lost_events")
        conn.commit()
        print("✓ Cleared old events from database")
    finally:
        conn.close()

def upsert_event(event_type, class_name, persistent_id, location, ts_iso, details=None):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        if event_type == 'start':
            cur.execute(
                """
                INSERT INTO lost_events (persistent_id, class_name, location, first_seen_ts, last_seen_ts, status, details)
                VALUES (?, ?, ?, ?, ?, 'inactive', ?)
                """,
                (persistent_id, class_name, location, ts_iso, ts_iso, details)
            )
        elif event_type == 'update':
            cur.execute(
                """
                UPDATE lost_events
                SET last_seen_ts = ?, details = COALESCE(?, details)
                WHERE status = 'inactive' AND persistent_id = ?
                """,
                (ts_iso, details, persistent_id)
            )
        elif event_type == 'resolve':
            cur.execute(
                """
                UPDATE lost_events
                SET last_seen_ts = ?, status = 'active', details = COALESCE(?, details)
                WHERE status = 'inactive' AND persistent_id = ?
                """,
                (ts_iso, details, persistent_id)
            )
        conn.commit()
    finally:
        conn.close()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>AI Based Lost Item Detector</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: linear-gradient(135deg, #1e3c72, #2a5298);
      margin: 0;
      padding: 0;
      color: white;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    h1 {
      text-align: center;
      margin-top: 30px;
      font-size: 2.5em;
      font-weight: bold;
      letter-spacing: 2px;
    }
    .button-container {
      margin-top: 30px;
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 20px;
    }
    button {
      padding: 15px 30px;
      font-size: 18px;
      font-weight: bold;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      color: white;
      background: linear-gradient(90deg, #ff512f, #dd2476);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    button:hover {
      transform: translateY(-3px);
      box-shadow: 0px 5px 15px rgba(0,0,0,0.3);
    }
    #log {
      margin-top: 40px;
      padding: 15px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      width: 80%;
      height: 300px;
      overflow-y: auto;
      white-space: pre-wrap;
      font-family: monospace;
      color: #e0e0e0;
      border: 1px solid rgba(255, 255, 255, 0.3);
    }
  </style>
</head>
<body>
  <h1>AI BASED LOST ITEM DETECTOR</h1>

  <div class="button-container">
    <button onclick="runMode('webcam')">🎥 Webcam Mode</button>
    <button onclick="runMode('video')">📹 Video Mode</button>
    <button onclick="runMode('list')">📂 List Available Videos</button>
    <button onclick="runMode('quit')">❌ Quit</button>
  </div>

  <div id="log">Logs will appear here...</div>

  <script>
    async function runMode(mode) {
      let videoPath = '';
      if (mode === 'video') {
        videoPath = prompt('Enter video file path:');
        if (!videoPath) return;
      }

      fetch(`/run/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ videoPath })
      });

      document.getElementById('log').innerHTML = '';
    }

    const evtSource = new EventSource('/logs');
    evtSource.onmessage = function(event) {
      const logDiv = document.getElementById('log');
      logDiv.innerHTML += event.data + "\\n";
      logDiv.scrollTop = logDiv.scrollHeight;
    };
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    try:
        return send_file('index.html')
    except Exception:
        return render_template_string(HTML_PAGE)

PUBLIC_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Lost & Found - Live</title>
  <style>
    body { font-family: Arial, sans-serif; background: #0b1020; color: #eee; margin: 0; padding: 20px; }
    h1 { margin: 0 0 10px 0; }
    .sub { color: #9aa; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; background: #151b2e; }
    th, td { padding: 10px; border-bottom: 1px solid #24304f; text-align: left; }
    th { background: #1b2340; }
    .pill { padding: 2px 8px; border-radius: 999px; font-weight: bold; font-size: 12px; }
    .pill.active { background: #2d7; color: #041; }
    .pill.inactive { background: #d72; color: #fff; }
  </style>
  <script>
    async function loadData() {
      const res = await fetch('/api/lost-events?status=all&limit=100');
      const data = await res.json();
      const tbody = document.getElementById('tbody');
      tbody.innerHTML = '';
      for (const row of data.items) {
        const tr = document.createElement('tr');
        const statusClass = row.status === 'active' ? 'active' : 'inactive';
        const statusText = row.status === 'active' ? 'ACTIVE' : 'INACTIVE';
        tr.innerHTML = `
          <td><span class="pill ${statusClass}">${statusText}</span></td>
          <td>${row.class_name}</td>
          <td>${row.location || '-'}</td>
          <td>${new Date(row.first_seen_ts).toLocaleString()}</td>
          <td>${new Date(row.last_seen_ts).toLocaleString()}</td>
          <td>#${row.persistent_id ?? ''}</td>
        `;
        tbody.appendChild(tr);
      }
    }
    setInterval(loadData, 5000);
    window.onload = loadData;
  </script>
</head>
<body>
  <h1>Lost & Found - Live</h1>
  <div class="sub">Auto-updates every 5 seconds. Share this page with students/staff.</div>
  <table>
    <thead>
      <tr>
        <th>Status</th>
        <th>Object</th>
        <th>Location</th>
        <th>First seen</th>
        <th>Last seen</th>
        <th>Item ID</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</body>
</html>
"""

@app.route("/public")
def public():
    return render_template_string(PUBLIC_PAGE)

@app.route("/run/<mode>", methods=["POST"])
def run_mode(mode):
    global process, log_buffer
    data = request.get_json(force=True)
    video_path = data.get("videoPath", "")

    if process and process.poll() is None:
        return "⚠ Process already running", 400

    if mode == "webcam":
        cmd = [sys.executable, "-u", "run_system.py", "--webcam"]
    elif mode == "video":
        if not video_path:
            return "Video path not provided", 400
        cmd = [sys.executable, "-u", "run_system.py", "--video", video_path]
    elif mode == "list":
        cmd = [sys.executable, "-u", "run_system.py", "--list-videos"]
    elif mode == "quit":
        if process:
            process.terminate()
            return "✅ Process terminated", 200
        return "No process to quit", 200
    else:
        return "Invalid mode", 400

    log_buffer = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')

    threading.Thread(target=stream_logs, daemon=True).start()
    return f"▶ Started {mode}", 200

def stream_logs():
    global process, log_buffer
    try:
        for line in iter(process.stdout.readline, ''):
            with lock:
                clean_line = line.rstrip()
                if clean_line:
                    log_buffer.append(clean_line)
        process.stdout.close()
        process.wait()
    except UnicodeDecodeError as e:
        print(f"Unicode decode error in stream_logs: {e}")
        try:
            process.stdout.close()
            process.wait()
        except:
            pass
    except Exception as e:
        print(f"Error in stream_logs: {e}")
        try:
            process.stdout.close()
            process.wait()
        except:
            pass

@app.route("/logs")
def logs():
    def event_stream():
        last_index = 0
        while True:
            try:
                with lock:
                    if last_index < len(log_buffer):
                        for i in range(last_index, len(log_buffer)):
                            log_line = log_buffer[i]
                            if isinstance(log_line, str):
                                clean_line = log_line.encode('utf-8', errors='replace').decode('utf-8')
                                yield f"data: {clean_line}\n\n"
                            else:
                                yield f"data: {str(log_line)}\n\n"
                        last_index = len(log_buffer)
                time.sleep(0.3)
            except Exception as e:
                print(f"Error in event_stream: {e}")
                yield f"data: Error in log streaming: {str(e)}\n\n"
                time.sleep(1)
    return Response(event_stream(), content_type="text/event-stream")

@app.route('/api/lost-events', methods=['GET', 'POST'])
def lost_events():
    if request.method == 'POST':
        data = request.get_json(force=True)
        event_type = data.get('event')
        class_name = data.get('className')
        persistent_id = data.get('persistentId')
        location = data.get('location')
        ts = data.get('timestamp')
        details = data.get('details')
        if not ts:
            ts = datetime.utcnow().isoformat() + 'Z'
        if not event_type or not class_name or persistent_id is None:
            return jsonify({'error': 'Missing required fields'}), 400
        try:
            upsert_event(event_type, class_name, int(persistent_id), location, ts, details)
            return jsonify({'ok': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        status = request.args.get('status', 'active')
        limit = int(request.args.get('limit', '50'))
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            if status == 'all':
                cur.execute("SELECT * FROM lost_events ORDER BY last_seen_ts DESC LIMIT ?", (limit,))
            else:
                cur.execute("SELECT * FROM lost_events WHERE status = ? ORDER BY last_seen_ts DESC LIMIT ?", (status, limit))
            rows = cur.fetchall()
            items = [dict(r) for r in rows]
            return jsonify({'items': items})
        finally:
            conn.close()

if __name__ == "__main__":
    init_db()
    clear_old_events()
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
