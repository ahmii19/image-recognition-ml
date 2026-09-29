# Google Colab GPU Remote Inference Setup Guide

**Document Version:** 1.0.0  
**Target Environment:** Google Colab (NVIDIA Tesla T4 / A100 GPU)  
**Client Environment:** Local Laptop (Next.js 14 Web Frontend + Antigravity IDE)

---

## Overview

This guide explains how to execute the compute-intensive computer vision pipeline (**OpenCLIP ViT-B/32, Google OWL-ViT, and Phase 6C Two-Stage Region Cascade**) on a free cloud GPU inside Google Colab, while keeping the Next.js web application running locally on your laptop.

```
+------------------------------------+           HTTPS (Cloudflare Tunnel)          +--------------------------------------+
|        LOCAL LAPTOP (Client)       | -------------------------------------------> |        GOOGLE COLAB (Server)         |
|  - Next.js UI (localhost:3000)     |                                              |  - FastAPI (Port 8000)               |
|  - .env.local -> Colab Public URL  | <------------------------------------------- |  - OpenCLIP + OWL-ViT on CUDA GPU    |
|  - Zero Heavy ML Memory on Laptop  |                 JSON Responses               |  - Phase 6C Batched Region Pipeline  |
+------------------------------------+                                              +--------------------------------------+
```

---

## Requirements

1. **Google Account:** Free Google account to access [Google Colab](https://colab.research.google.com).
2. **GPU Runtime:** Standard Colab T4 GPU (16 GB VRAM).
3. **Project Source:** Access to the project Git repository or uploaded project directory.
4. **Local Development Environment:** Node.js (v18+) for running the Next.js frontend on your laptop.

---

## Step-by-Step Setup Procedure

### Step 1: Open the Colab Notebook
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **File -> Upload notebook** and upload [notebooks/PHASE_GPU_C_COLAB_SETUP.ipynb](file:///d:/AHMED%20PROJECTS/ml%20project/notebooks/PHASE_GPU_C_COLAB_SETUP.ipynb).

### Step 2: Select GPU Runtime
1. In the Colab top menu, click **Runtime -> Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or any available GPU).
3. Click **Save**.

### Step 3: Run Setup & Verification Cells (Cells 1 to 5)
1. **Run Cell 1 (GPU Diagnostics):** Confirms an NVIDIA GPU is attached (prints GPU model name, CUDA version, and ~15–16 GB total VRAM).
2. **Run Cell 2 (Repository Setup):** Clones or links your repository to `/content/ml-project` and adds it to Python's `sys.path`.
3. **Run Cell 3 (Dependencies):** Installs `open-clip-torch`, `transformers`, `fastapi`, and `uvicorn`.
4. **Run Cell 4 (Environment Config):** Sets `ML_DEVICE=cuda` and verifies device resolution.
5. **Run Cell 5 (Model Verification):** Loads OpenCLIP and OWL-ViT into GPU VRAM and performs a warmup inference pass.

### Step 4: Start FastAPI Server & Cloudflare Tunnel (Cells 6 to 8)
1. **Run Cell 6 (Start FastAPI):** Launches the FastAPI server in the background on port 8000 and waits for health readiness.
2. **Run Cell 7 (Health Check):** Queries `/api/v1/health` and `/api/v1/models` to confirm active CUDA serving.
3. **Run Cell 8 (Tunnel Startup):** Launches an encrypted Cloudflare Tunnel and prints the public API URL:
   ```
   ======================================================================
   PUBLIC DEVELOPMENT API TUNNEL IS LIVE
   ======================================================================
   
   COLAB_API_URL = https://random-subdomain.trycloudflare.com
   ```

### Step 5: Connect Local Next.js Frontend (On Your Laptop)
1. Open [frontend/.env.local](file:///d:/AHMED%20PROJECTS/ml%20project/frontend/.env.local) on your laptop.
2. Update `NEXT_PUBLIC_API_BASE_URL` with the generated URL:
   ```bash
   # frontend/.env.local
   NEXT_PUBLIC_API_BASE_URL=https://random-subdomain.trycloudflare.com
   ```
3. Restart your Next.js development server:
   ```powershell
   cd frontend
   npm run dev
   ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser.
5. Upload an image and test:
   - **Zero-Shot Classification** (OpenCLIP on CUDA)
   - **Open-Vocabulary Object Detection** (OWL-ViT on CUDA)
   - **Grounded Region Intelligence** (Phase 6C Two-Stage on CUDA)

---

## Local CPU Fallback (Switching Back to Localhost)

If you disconnect from Colab or want to run offline locally on CPU:

1. Edit [frontend/.env.local](file:///d:/AHMED%20PROJECTS/ml%20project/frontend/.env.local):
   ```bash
   NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
   ```
2. Start your local FastAPI backend:
   ```powershell
   $env:ML_DEVICE="cpu"
   uvicorn src.api.main:app --port 8000
   ```
3. Refresh your browser at [http://localhost:3000](http://localhost:3000).

---

## Important Colab Limitations

> [!WARNING]
> **Google Colab is a development/experimentation platform, NOT a permanent production server.**

1. **Session Timeout:** Free Colab sessions disconnect after ~15–30 minutes of inactivity and terminate after a maximum of 12 hours.
2. **Ephemeral Disk:** Any downloaded models or logs in Colab are reset when the runtime terminates.
3. **Dynamic Tunnel URLs:** Each time you re-run Cell 8, a new Cloudflare tunnel URL is generated, requiring a quick update in `frontend/.env.local`.
4. **Security Notice:** The generated tunnel URL is publicly reachable over the internet. Never expose sensitive private data or secret credentials through the development tunnel.

---

## Stopping & Releasing Colab Resources

When your testing session is finished, run **Cell 11** in the notebook to:
1. Terminate the background FastAPI server and Cloudflare tunnel.
2. Unload OWL-ViT and call `torch.cuda.empty_cache()` to free all GPU VRAM.
3. Disconnect runtime via Colab menu: **Runtime -> Disconnect and delete runtime**.
