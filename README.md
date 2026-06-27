# LeLamp Mini: Edge AI & Asynchronous Robotics Backend

LeLamp is an intelligent, privacy-first conversational robotic desk lamp architecture. The system combines local, edge-based computer vision tracking with local Large Language Model (LLM) inference to build a persistent spatial memory of its physical environment. 

The entire framework runs completely offline on local hardware, keeping data entirely private.

---

## Key Architectural Features

* **Dual-Thread Perception Engine:** Utilizes MediaPipe Face Mesh in a dedicated background worker thread to process real-time 3D head-pose estimation and gaze-engagement tracking.
* **Live Object Detection Pipeline:** Uses an ultra-lightweight YOLOv8 Nano model to dynamically scan the workspace for everyday objects (phones, keys, cups) when user engagement is detected.
* **Persistent Spatial Memory Storage:** Automatically writes real-time normalized coordinate logs into a thread-safe, local SQLite database on behavioral state changes.
* **Tool-Augmented Retrieval (RAG):** Hooks a locally hosted `llama3.2:1b` model via Ollama into the database telemetry. The system dynamically formats user prompts into strict data readouts to reliably serve real-time coordinate locations without guardrail false-positives.
* **Asynchronous Central Controller:** A core `asyncio` finite state machine (FSM) acting at 20Hz coordinates sensory input loops, input logging, and local inference cleanly without UI blocking.

---

## Technology Stack

* **Language:** Python 3.12
* **Vision & Alignment:** OpenCV, MediaPipe, Ultralytics (YOLOv8)
* **Local Inference:** Ollama Engine (`llama3.2:1b`)
* **Database:** SQLite3
* **Concurrency:** Asyncio, Concurrent.Futures (ThreadPoolExecutor)

---

## Quick Start

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.com) installed and the background service running:
```bash
ollama run llama3.2:1b
Ensure you have [Ollama](https://ollama.com) installed and the background service running:
```bash
ollama run llama3.2:1b
