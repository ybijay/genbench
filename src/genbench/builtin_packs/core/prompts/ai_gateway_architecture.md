Design a production-grade local AI inference gateway architecture connecting multiple developer workstations (running VS Code coding agents) to a single Linux server equipped with one 24GB VRAM GPU (NVIDIA Quadro P6000) running Ollama. Your architecture design must cover:
1. Request queueing, concurrency control, and backpressure to prevent VRAM OOM thrashing.
2. Model hot-swapping vs. KV-cache context pinning trade-offs (especially for 32K-64K contexts).
3. Time-To-First-Token (TTFT) & prompt-cache optimization for multi-turn IDE coding agents.
4. Health checks, circuit breaking, and fallback routing.
Include a clear ASCII architecture diagram and a markdown comparison table of design trade-offs.
