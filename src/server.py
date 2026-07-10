"""
Chat playground server.

- GET  /                      -> the chat UI (static/index.html)
- WS   /ws/run                -> client sends {"goal": "...", "self_correction": true},
                                   server streams agent events live as JSON frames,
                                   then persists the finished RunLog to logs/<run_id>.json
- GET  /api/logs               -> list of past run ids (+ short metadata)
- GET  /logs/{run_id}          -> human-readable HTML render of a stored run log

The agent itself is synchronous (it runs real subprocess tool calls), so we
run it in a background thread per websocket connection and forward events
into the asyncio loop via an asyncio.Queue, which is the standard pattern
for bridging sync work into an async websocket without blocking the server.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agent import Agent
from .log_viewer import list_run_ids, load_run_log, persist_run_log, render_run_log_html

app = FastAPI(title="Heva Self-Correcting Agent Playground")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/logs")
def api_logs():
    ids = list_run_ids()
    out = []
    for rid in ids[::-1]:
        try:
            rl = load_run_log(rid)
            out.append({
                "run_id": rl.run_id,
                "goal": rl.goal,
                "self_correction_enabled": rl.self_correction_enabled,
                "completed": rl.completed,
                "self_corrections": rl.self_corrections,
                "unresolved": len(rl.unresolved_subtasks),
            })
        except Exception:
            continue
    return JSONResponse(out)


@app.get("/logs/{run_id}", response_class=HTMLResponse)
def view_log(run_id: str):
    rl = load_run_log(run_id)
    return render_run_log_html(rl)


@app.websocket("/ws/run")
async def ws_run(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        req = json.loads(raw)
        goal = req.get("goal", "").strip()
        self_correction = bool(req.get("self_correction", True))
        if not goal:
            await websocket.send_json({"type": "error", "message": "goal is required"})
            await websocket.close()
            return

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_event(event: dict):
            # called from the worker thread; hop back onto the event loop safely
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def worker():
            agent = Agent()
            try:
                run_log = agent.run(goal, self_correction=self_correction, on_event=on_event)
                persist_run_log(run_log)
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "__done__"})

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            event = await queue.get()
            if event.get("type") == "__done__":
                break
            await websocket.send_json(event)

        await websocket.close()
    except WebSocketDisconnect:
        pass
