# Lark 🎵

**Forecasting on a Lark.**

> *"I did it on a lark." — Doing something for fun and adventure.*

Lark is a **brutally efficient** Go inference server built for the **High-Cardinality** problem. It allows you to run thousands of tiny time-series models on a single private machine (or edge device) without the massive overhead of Docker or Kubernetes.

[![Go Version](https://img.shields.io/github/go-mod/go-version/yourname/lark)](https://go.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🐥 The Problem: The "Density Gap"

Standard MLOps tools (Ray, TorchServe, K8s) prioritize **Isolation**. They wrap every model in a heavy container or Python process.
* **The Result:** Running 1,000 models requires ~100GB of RAM.

Lark prioritizes **Density**. It treats models like data, not services.
* **The Result:** Lark runs 1,000 models in ~2GB of RAM on a standard laptop.

## 🎯 Architecture: "The Flock"

Lark uses a biological metaphor to manage resources efficiently:

1.  **The Nest (Queue):** A buffered channel that protects the server from overload. If the nest is full, Lark chirps `503 Service Unavailable` instantly (Backpressure).
2.  **The Flock (Worker Pool):** A fixed number of Goroutines (bound to CPU cores) that handle inference.
3.  **The Breeder (Polyglot Training):** A "Sidecar" interface that lets you train models using **Python**, **Julia**, or **Rust**, while Go manages the lifecycle and I/O.



## ⚡ Benchmarks (Projected)

| Metric | Docker/K8s (Python) | Lark (Go) |
| :--- | :--- | :--- |
| **Unit of Concurrency** | OS Process / Container | Goroutine |
| **Idle RAM (1k Models)** | ~100 GB (Crash) | ~2 GB (Stable) |
| **Startup Time** | ~3.0s (Container) | ~0.01s (JIT Load) |
| **Philosophy** | "Isolation First" | "Density First" |

## 🛠️ Quick Start

### Prerequisites
* Go 1.23+
* Python 3.10+ (Optional, for training sidecar)

### Running the Flock
```bash
# Start Lark with 8 workers (birds) and port 8080
go run cmd/lark/main.go --birds 8 --port 8080
Usage (API)
1. Predict (Inference)

Bash
curl -X POST http://localhost:8080/chirp \
  -H "Content-Type: application/json" \
  -d '{"model_id": "sensor_99", "data": [0.1, 0.5, 0.8]}'
2. Breed (Train New Model)

Bash
curl -X POST http://localhost:8080/breed \
  -H "Content-Type: application/json" \
  -d '{"model_id": "sensor_99", "dataset": "data/sensor_99.csv"}'
🗺️ Roadmap to v1.0
[x] Core: Worker Pool ("The Flock") & Queue ("The Nest")

[ ] Engine: CGO Bindings for ONNX Runtime

[ ] Polyglot: Python Sidecar for Training

[ ] Memory: LRU Caching for "Hot" Models

[ ] Edge: WASM/WASI Support for safe model breeding

Built with ❤️ in Go.
