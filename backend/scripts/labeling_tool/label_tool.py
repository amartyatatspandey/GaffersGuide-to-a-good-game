#!/usr/bin/env python3
"""
label_tool.py — Minimal self-contained local HTML labeling tool for formation clips.
================================================================================

This script launches a small local HTTP server on a free port on localhost.
It serves a premium web UI to label the 46 generated clips from:
    backend/data/real_eval_set/manifest.csv

No external python dependencies are required. Just run:
    python3 backend/scripts/labeling_tool/label_tool.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import socket
import urllib.parse
import http.server
import socketserver

# ── Path Setup ──────────────────────────────────────────────────────────────
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SCRIPT_DIR is backend/scripts/labeling_tool
# PROJECT_ROOT is GaffersGuide-to-a-good-game (3 levels up)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

parser = argparse.ArgumentParser(description="Gaffer's Guide - Tactical Formation Labeling Tool")
parser.add_argument("--dataset", type=str, default="real_eval_set", choices=["real_eval_set", "cnn_baseline"], help="Dataset to label")
args, unknown = parser.parse_known_args()

DATASET = args.dataset
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "backend", "data", DATASET, "manifest.csv")
PREVIEWS_DIR = os.path.join(PROJECT_ROOT, "backend", "data", DATASET, "previews")
HEATMAPS_DIR = os.path.join(PROJECT_ROOT, "backend", "data", DATASET, "heatmaps")

# ── Configuration ───────────────────────────────────────────────────────────
CLASSES = [
    "4-3-3 High Press",
    "4-2-3-1 Mid Block",
    "3-5-2 Wing Backs",
    "Double Pivot",
    "Inverted Full Backs",
    "UNCLEAR / SKIP"
]

# Check that manifest file exists
if not os.path.exists(MANIFEST_PATH):
    print(f"[-] ERROR: Manifest CSV file not found at: {MANIFEST_PATH}", file=sys.stderr)
    print(f"[-] Please generate data first for dataset: {DATASET}", file=sys.stderr)
    sys.exit(1)

# ── Data Operations ─────────────────────────────────────────────────────────
def load_manifest() -> list[dict[str, str]]:
    clips = []
    with open(MANIFEST_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append({
                "clip_id": row["clip_id"],
                "video_source": row["video_source"],
                "frame_range": row["frame_range"],
                "heatmap_path": row["heatmap_path"],
                "label": row["label"]
            })
    return clips

def save_manifest(clips: list[dict[str, str]]) -> None:
    with open(MANIFEST_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "video_source", "frame_range", "heatmap_path", "label"])
        writer.writeheader()
        writer.writerows(clips)

# ── HTML Template ───────────────────────────────────────────────────────────
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gaffer's Guide - Tactical Formation Labeling Tool</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 28, 45, 0.6);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.15);
            --success: #34d399;
            --success-glow: rgba(52, 211, 153, 0.15);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            
            /* Class Colors */
            --color-433: #a78bfa;     /* Violet */
            --color-4231: #818cf8;    /* Indigo */
            --color-352: #38bdf8;     /* Cyan */
            --color-dblpiv: #2dd4bf;  /* Teal */
            --color-ifb: #60a5fa;     /* Blue */
            --color-skip: #64748b;    /* Slate */
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(129, 140, 248, 0.08) 0px, transparent 50%);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            padding: 2rem 1.5rem;
            overflow-x: hidden;
        }
        
        .container {
            width: 100%;
            max-width: 1200px;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
        }
        
        .logo-section h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.025em;
        }
        
        .logo-section p {
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }
        
        .progress-container {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 0.5rem;
            width: 250px;
        }
        
        .progress-text {
            font-size: 0.875rem;
            color: var(--text-muted);
            font-weight: 500;
        }
        
        .progress-text span {
            color: var(--text-main);
            font-weight: 700;
        }
        
        .progress-bar-bg {
            width: 100%;
            height: 8px;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .progress-bar-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #38bdf8 0%, #34d399 100%);
            border-radius: 4px;
            transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
        }
        
        .main-card {
            background-color: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        
        .viewer-grid {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 1.5rem;
            align-items: start;
        }
        
        @media (max-width: 900px) {
            .viewer-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .image-box {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            background-color: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 1rem;
        }
        
        .image-box-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .image-container {
            position: relative;
            width: 100%;
            aspect-ratio: 16/9;
            background-color: #05070c;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.6);
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            opacity: 0;
            transition: opacity 0.15s ease-in-out;
        }
        
        .image-container img.loaded {
            opacity: 1;
        }
        
        .image-loader {
            position: absolute;
            width: 32px;
            height: 32px;
            border: 3px solid rgba(56, 189, 248, 0.1);
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 12px;
            padding: 1.25rem;
        }
        
        .meta-item {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        
        .meta-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .meta-value {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-main);
        }
        
        .meta-value.label-badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.85rem;
            background-color: rgba(255, 255, 255, 0.05);
            width: fit-content;
        }
        
        .labeling-section {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .section-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }
        
        .buttons-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.75rem;
        }
        
        .label-btn {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 1rem;
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .label-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            background-color: rgba(255, 255, 255, 0.06);
        }
        
        .label-btn:active {
            transform: translateY(0);
        }
        
        .shortcut-badge {
            background-color: rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            font-family: monospace;
        }
        
        /* Button colors */
        .btn-433 { border-left: 4px solid var(--color-433); }
        .btn-433:hover { border-color: var(--color-433); background-color: rgba(167, 139, 250, 0.08); box-shadow: 0 0 15px rgba(167, 139, 250, 0.15); }
        .btn-433 .shortcut-badge { color: var(--color-433); border-color: rgba(167, 139, 250, 0.3); }
        
        .btn-4231 { border-left: 4px solid var(--color-4231); }
        .btn-4231:hover { border-color: var(--color-4231); background-color: rgba(129, 140, 248, 0.08); box-shadow: 0 0 15px rgba(129, 140, 248, 0.15); }
        .btn-4231 .shortcut-badge { color: var(--color-4231); border-color: rgba(129, 140, 248, 0.3); }
        
        .btn-352 { border-left: 4px solid var(--color-352); }
        .btn-352:hover { border-color: var(--color-352); background-color: rgba(56, 189, 248, 0.08); box-shadow: 0 0 15px rgba(56, 189, 248, 0.15); }
        .btn-352 .shortcut-badge { color: var(--color-352); border-color: rgba(56, 189, 248, 0.3); }
        
        .btn-dblpiv { border-left: 4px solid var(--color-dblpiv); }
        .btn-dblpiv:hover { border-color: var(--color-dblpiv); background-color: rgba(45, 212, 191, 0.08); box-shadow: 0 0 15px rgba(45, 212, 191, 0.15); }
        .btn-dblpiv .shortcut-badge { color: var(--color-dblpiv); border-color: rgba(45, 212, 191, 0.3); }
        
        .btn-ifb { border-left: 4px solid var(--color-ifb); }
        .btn-ifb:hover { border-color: var(--color-ifb); background-color: rgba(96, 165, 250, 0.08); box-shadow: 0 0 15px rgba(96, 165, 250, 0.15); }
        .btn-ifb .shortcut-badge { color: var(--color-ifb); border-color: rgba(96, 165, 250, 0.3); }
        
        .btn-skip { border-left: 4px solid var(--color-skip); }
        .btn-skip:hover { border-color: var(--color-skip); background-color: rgba(100, 116, 139, 0.08); box-shadow: 0 0 15px rgba(100, 116, 139, 0.15); }
        .btn-skip .shortcut-badge { color: var(--color-skip); border-color: rgba(100, 116, 139, 0.3); }

        .navigation-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid var(--border-color);
            padding-top: 1.5rem;
            margin-top: 0.5rem;
        }
        
        .nav-btn {
            background-color: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-main);
            border-radius: 8px;
            padding: 0.75rem 1.25rem;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s ease;
        }
        
        .nav-btn:hover:not(:disabled) {
            background-color: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .nav-btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }
        
        .footer-info {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 1rem;
            display: flex;
            justify-content: center;
            gap: 2rem;
        }
        
        .kbd-guide {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .kbd-key {
            background-color: rgba(255, 255, 255, 0.1);
            color: var(--text-main);
            padding: 0.1rem 0.3rem;
            border-radius: 3px;
            font-size: 0.7rem;
            font-family: monospace;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        /* Status animations */
        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background-color: #10b981;
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            z-index: 100;
        }
        
        .toast.show {
            transform: translateY(0);
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-section">
                <h1>GAFFER'S GUIDE</h1>
                <p>Formation Labeling Tool • Manual Evaluator</p>
            </div>
            <div class="progress-container">
                <div class="progress-text">Progress: <span id="progress-val">0 / 0</span> labeled (<span id="progress-pct">0%</span>)</div>
                <div class="progress-bar-bg">
                    <div id="progress-fill" class="progress-bar-fill"></div>
                </div>
            </div>
        </header>
        
        <main class="main-card">
            <div class="viewer-grid">
                <div class="image-box">
                    <div class="image-box-title">
                        Camera Preview (640x360)
                    </div>
                    <div class="image-container">
                        <div id="preview-loader" class="image-loader"></div>
                        <img id="preview-img" alt="Camera Preview" src="" onload="imageLoaded('preview')">
                    </div>
                </div>
                
                <div class="image-box">
                    <div class="image-box-title">
                        Gaussian Position Heatmap
                    </div>
                    <div class="image-container">
                        <div id="heatmap-loader" class="image-loader"></div>
                        <img id="heatmap-img" alt="Tactical Heatmap" src="" onload="imageLoaded('heatmap')">
                    </div>
                </div>
            </div>
            
            <div class="meta-grid">
                <div class="meta-item">
                    <span class="meta-label">Clip ID</span>
                    <span class="meta-value" id="meta-clip-id">-</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Frame Range</span>
                    <span class="meta-value" id="meta-frame-range">-</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Video Source</span>
                    <span class="meta-value" style="font-family: monospace; font-size: 0.9rem;" id="meta-video-source">-</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Current Label</span>
                    <span class="meta-value label-badge" id="meta-label">-</span>
                </div>
            </div>
            
            <div class="labeling-section">
                <h3 class="section-title">Assign Tactical Formation</h3>
                <div class="buttons-grid" id="buttons-container">
                    <!-- Dynamic buttons -->
                </div>
            </div>
            
            <div class="navigation-row">
                <button id="btn-back" class="nav-btn" onclick="goBack()" disabled>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
                    Back (Left Arrow)
                </button>
                <button id="btn-next" class="nav-btn" onclick="goNext()">
                    Skip / Next (Right Arrow)
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                </button>
            </div>
        </main>
        
        <div class="footer-info">
            <span class="kbd-guide">
                Labeling: <span class="kbd-key">1</span> to <span class="kbd-key">6</span> keys
            </span>
            <span class="kbd-guide">
                Back: <span class="kbd-key">← Left Arrow</span> / <span class="kbd-key">Backspace</span>
            </span>
            <span class="kbd-guide">
                Next/Skip: <span class="kbd-key">→ Right Arrow</span>
            </span>
        </div>
    </div>
    
    <div id="toast" class="toast">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
        <span id="toast-text">Label saved successfully!</span>
    </div>
    
    <script>
        let state = {
            clips: [],
            classes: [],
            currentIndex: 0
        };
        
        const btnClasses = [
            'btn-433',
            'btn-4231',
            'btn-352',
            'btn-dblpiv',
            'btn-ifb',
            'btn-skip'
        ];
        
        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                state.clips = data.clips;
                state.classes = data.classes;
                state.dataset = data.dataset || 'real_eval_set';
                
                // Find first unlabeled clip
                const firstUnlabeled = state.clips.findIndex(c => c.label === 'UNLABELED');
                state.currentIndex = firstUnlabeled >= 0 ? firstUnlabeled : 0;
                
                renderButtons();
                showClip(state.currentIndex);
            } catch (err) {
                console.error("Error loading application state:", err);
                alert("Could not load manifest. Make sure the server is running.");
            }
        }
        
        function renderButtons() {
            const container = document.getElementById('buttons-container');
            container.innerHTML = '';
            
            state.classes.forEach((className, idx) => {
                const btnClass = btnClasses[idx] || 'btn-skip';
                const btn = document.createElement('button');
                btn.className = `label-btn ${btnClass}`;
                btn.innerHTML = `
                    <span>${className}</span>
                    <span class="shortcut-badge">${idx + 1}</span>
                `;
                btn.onclick = () => submitLabel(className);
                container.appendChild(btn);
            });
        }
        
        function showClip(index) {
            if (index < 0 || index >= state.clips.length) return;
            state.currentIndex = index;
            
            const clip = state.clips[index];
            
            // Set image source and trigger loader
            document.getElementById('preview-loader').style.display = 'block';
            document.getElementById('heatmap-loader').style.display = 'block';
            
            const previewImg = document.getElementById('preview-img');
            const heatmapImg = document.getElementById('heatmap-img');
            
            previewImg.classList.remove('loaded');
            heatmapImg.classList.remove('loaded');
            
            // Previews are under /backend/data/{dataset}/previews/clip_xxx_preview.jpg
            previewImg.src = `/backend/data/${state.dataset}/previews/${clip.clip_id}_preview.jpg`;
            
            // Heatmap path is defined in clip.heatmap_path (e.g. backend/data/real_eval_set/heatmaps/clip_xxx_heatmap.png)
            heatmapImg.src = `/${clip.heatmap_path}`;
            
            // Set details
            document.getElementById('meta-clip-id').innerText = clip.clip_id;
            document.getElementById('meta-frame-range').innerText = clip.frame_range;
            document.getElementById('meta-video-source').innerText = clip.video_source;
            
            const labelBadge = document.getElementById('meta-label');
            labelBadge.innerText = clip.label;
            
            // Apply subtle visual styling to badge depending on whether it's labeled
            if (clip.label === 'UNLABELED') {
                labelBadge.style.color = '#e2e8f0';
                labelBadge.style.borderColor = 'rgba(255,255,255,0.1)';
                labelBadge.style.backgroundColor = 'rgba(255,255,255,0.05)';
            } else {
                labelBadge.style.color = '#34d399';
                labelBadge.style.borderColor = 'rgba(52,211,153,0.3)';
                labelBadge.style.backgroundColor = 'rgba(52,211,153,0.1)';
            }
            
            // Set progress
            const labeledCount = state.clips.filter(c => c.label !== 'UNLABELED').length;
            const total = state.clips.length;
            const pct = total > 0 ? Math.round((labeledCount / total) * 100) : 0;
            
            document.getElementById('progress-val').innerText = `${labeledCount} / ${total}`;
            document.getElementById('progress-pct').innerText = `${pct}%`;
            document.getElementById('progress-fill').style.width = `${pct}%`;
            
            // Set buttons state
            document.getElementById('btn-back').disabled = (index === 0);
        }
        
        function imageLoaded(type) {
            document.getElementById(`${type}-loader`).style.display = 'none';
            document.getElementById(`${type}-img`).classList.add('loaded');
        }
        
        async function submitLabel(labelName) {
            const clip = state.clips[state.currentIndex];
            try {
                const res = await fetch('/api/label', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ clip_id: clip.clip_id, label: labelName })
                });
                
                if (res.ok) {
                    clip.label = labelName;
                    showToast(`Saved as "${labelName}"`);
                    
                    // Advance to next unlabeled
                    let nextIndex = state.currentIndex + 1;
                    let found = false;
                    
                    // Look forward
                    for (let i = nextIndex; i < state.clips.length; i++) {
                        if (state.clips[i].label === 'UNLABELED') {
                            nextIndex = i;
                            found = true;
                            break;
                        }
                    }
                    
                    // Look backward from start if not found
                    if (!found) {
                        for (let i = 0; i < state.currentIndex; i++) {
                            if (state.clips[i].label === 'UNLABELED') {
                                nextIndex = i;
                                found = true;
                                break;
                            }
                        }
                    }
                    
                    if (found) {
                        showClip(nextIndex);
                    } else {
                        // All labeled! Refresh current view
                        showClip(state.currentIndex);
                        setTimeout(() => {
                            showCompletionToast();
                        }, 200);
                    }
                } else {
                    alert("Error saving label. Please try again.");
                }
            } catch (err) {
                console.error("Error submitting label:", err);
                alert("Failed to reach server to save label.");
            }
        }
        
        function goBack() {
            if (state.currentIndex > 0) {
                showClip(state.currentIndex - 1);
            }
        }
        
        function goNext() {
            if (state.currentIndex < state.clips.length - 1) {
                showClip(state.currentIndex + 1);
            } else {
                // Loop back to start
                showClip(0);
            }
        }
        
        function showToast(msg) {
            const toast = document.getElementById('toast');
            document.getElementById('toast-text').innerText = msg;
            toast.className = 'toast show';
            setTimeout(() => {
                toast.className = 'toast';
            }, 1200);
        }
        
        function showCompletionToast() {
            alert("All clips labeled! Verification set is complete. Thank you!");
        }
        
        // Keyboard Shortcuts
        document.addEventListener('keydown', (e) => {
            // Prevent shortcuts when typing in inputs
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            const key = e.key;
            
            // Mapped keys 1-6
            if (key >= '1' && key <= '6') {
                const idx = parseInt(key) - 1;
                if (idx < state.classes.length) {
                    submitLabel(state.classes[idx]);
                }
            }
            
            // Navigation
            if (key === 'ArrowLeft' || key === 'Backspace') {
                goBack();
            } else if (key === 'ArrowRight') {
                goNext();
            }
        });
        
        // Initial state load
        fetchState();
    </script>
</body>
</html>
"""

# ── Server Implementation ───────────────────────────────────────────────────
class LabelingToolHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy assets and state requests to keep the server log clean
        if "GET /backend/data/" in args[0] or "GET /api/state" in args[0]:
            return
        super().log_message(format, *args)

    def do_GET(self):
        # 1. Serve UI Main Page
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
            return
            
        # 2. Serve Application State JSON
        elif self.path == '/api/state':
            try:
                clips = load_manifest()
                response_data = {
                    "clips": clips,
                    "classes": CLASSES,
                    "dataset": DATASET
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                return
            except Exception as e:
                self.send_error(500, f'Error reading manifest: {str(e)}')
                return
                
        # 3. Serve Static Media Files
        elif self.path.startswith('/backend/data/'):
            decoded_path = urllib.parse.unquote(self.path)
            rel_path = decoded_path.lstrip('/')
            abs_path = os.path.abspath(os.path.join(PROJECT_ROOT, rel_path))
            
            # Security verification
            allowed_root = os.path.abspath(os.path.join(PROJECT_ROOT, 'backend', 'data', DATASET))
            if abs_path.startswith(allowed_root):
                if os.path.exists(abs_path) and os.path.isfile(abs_path):
                    self.send_response(200)
                    if abs_path.endswith('.png'):
                        self.send_header('Content-type', 'image/png')
                    elif abs_path.endswith('.jpg') or abs_path.endswith('.jpeg'):
                        self.send_header('Content-type', 'image/jpeg')
                    else:
                        self.send_header('Content-type', 'application/octet-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    with open(abs_path, 'rb') as f:
                        self.wfile.write(f.read())
                    return
            
            self.send_error(404, 'File not found')
            return
            
        # 4. Fallback Not Found
        else:
            self.send_error(404, 'Not Found')
            return

    def do_POST(self):
        if self.path == '/api/label':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                clip_id = data.get('clip_id')
                label = data.get('label')
                
                if not clip_id or not label:
                    self.send_error(400, 'Missing clip_id or label')
                    return
                    
                if label not in CLASSES:
                    self.send_error(400, f'Invalid label: {label}')
                    return
                    
                clips = load_manifest()
                updated = False
                for clip in clips:
                    if clip['clip_id'] == clip_id:
                        clip['label'] = label
                        updated = True
                        break
                        
                if not updated:
                    self.send_error(404, 'Clip not found')
                    return
                    
                save_manifest(clips)
                print(f"[+] Labeled {clip_id} as '{label}' (manifest saved)")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
                return
                
            except Exception as e:
                self.send_error(500, f'Internal error: {str(e)}')
                return
        else:
            self.send_error(404, 'Not Found')
            return

# ── Entry Point ─────────────────────────────────────────────────────────────
def main() -> None:
    PORT = 5001
    server = None
    while True:
        try:
            # Bind to localhost / 127.0.0.1 for local usage
            server = socketserver.TCPServer(('127.0.0.1', PORT), LabelingToolHandler)
            break
        except OSError:
            # Port busy, try next port
            PORT += 1
            if PORT > 6000:
                print("[-] ERROR: Could not find a free port to bind to.", file=sys.stderr)
                sys.exit(1)

    print("=" * 80)
    print("   GAFFER'S GUIDE - TACTICAL FORMATION LABELING TOOL")
    print("=" * 80)
    print(f"   Server started successfully on localhost.")
    print(f"   Open your browser and navigate to: http://127.0.0.1:{PORT}")
    print("=" * 80)
    print("   Keyboard shortcuts:")
    print("     • [1] - [6] : Choose formation label")
    print("     • [Left Arrow] / [Backspace] : Go back one clip")
    print("     • [Right Arrow] : Skip / Next clip")
    print("=" * 80)
    print("   Logs:")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Server stopped by user. Goodbye!")
        server.server_close()
        sys.exit(0)

if __name__ == "__main__":
    main()
