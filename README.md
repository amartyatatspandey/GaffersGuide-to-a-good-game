

⸻

📘 Gaffer’s Guide — Open-Source Football Tactical Intelligence (Preliminary Release)

Gaffer’s Guide is an early-stage open-source tactical analysis platform built on top of SkillCorner spatio-temporal tracking data.
This project demonstrates how raw player tracking coordinates can be transformed into meaningful tactical insights through automated models and an interactive web dashboard.

This repository is the Phase I release of the platform — focusing on foundational analytics such as defensive line behaviour, physical performance, and possession centroid mapping. Future phases will expand into advanced tactics, pitch control, and deep-learning-based recognition of match patterns.

⸻

⚽ Features (Current Release)

🟦 Model 1 — Defensive Line Height (0–100 Tactical Behaviour Metric)
	•	Computes the defensive line depth of both teams for every frame (~25 FPS)
	•	Normalizes values to a 0–100 scale
	•	Produces colour-coded plots (Deep / Balanced / High / Aggressive)
	•	Includes quarter & halftime markers for match context

🟩 Model 2 — Player Physical Performance
	•	Total distance covered (km)
	•	Sprinting duration (s)
	•	Fatigue estimation (%)
	•	Auto-generated per-player comparison bar charts
	•	Handles missing or incomplete tracking data gracefully

🟥 Model 3 — Possession Centroid Heatmaps
	•	Calculates team possession frame-by-frame
	•	Computes centroids of where a team held the ball
	•	Generates a heatmap over the pitch
	•	Produces a 3×3 tactical grid with possession percentages

🖥 Interactive Flask Web App
	•	Auto-discovers matches on your machine
	•	Clean UI for browsing match metadata
	•	One-click execution of Models 1–3
	•	Safe file-serving (prevents path traversal vulnerabilities)

⸻

📂 Required Data Format (SkillCorner OpenData)

To use this tool, ensure you have SkillCorner-format match folders on your machine.

Expected structure:

opendata/data/matches/<match_id>/
│
├── match.json                           # match metadata
├── dynamic_events_<match_id>.csv        # event dataset
└── tracking_extrapolated.jsonl          # tracking coordinates

Example tracking JSONL line:

{
  "frame": 1520,
  "player_data": [
    {"player_id": 101, "team_id": 1, "x": -12.3, "y": 23.8},
    {"player_id": 102, "team_id": 1, "x": -20.5, "y": -8.1}
  ]
}

Example dynamic events CSV:

frame_start,team_id,player_id,event_name,...
2211,1,101,pass,...
2212,2,205,tackle,...

Your repository will not run without these files.

⸻

🚀 How to Run the Web App

1. Clone this repo

git clone <your-repository-url>
cd <repository-folder>

2. Place SkillCorner match folders

Place them here:

./opendata/data/matches/<match_id>/

3. Create a virtual environment (recommended)

python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

4. Install dependencies

pip install -r requirements.txt

If no requirements file exists:

pip install flask numpy pandas matplotlib

5. Start the Flask server

python app.py

Then open the web interface:

http://127.0.0.1:5000


⸻

🧭 Using the Platform

Home Page

Basic overview and navigation.

Matches Page

Automatically lists all match folders found in:

opendata/data/matches/

Match Detail Page

Shows metadata and provides buttons to run the three models.

Model Result Pages

Each model generates plots, stats, and downloadable outputs.

⸻

🔧 Tech Stack
	•	Backend: Python, Flask
	•	Data Handling: Pandas, NumPy
	•	Visualization: Matplotlib
	•	Data Source: SkillCorner OpenData
	•	Frontend: HTML/Jinja2 Templates

⸻

🔮 Roadmap (Future Work)

This is the Preliminary Stage of the full tactical system. Planned upcoming features:
	•	Pitch Control Model
	•	Team Compactness, Team Surface Area, &  Defensive Block Height
	•	Tactical Event Detection (pressing, counter-pressing, overloads)
	•	Pass network analytics
	•	Momentum modelling
	•	Machine learning classification of tactical patterns
	•	Visual dashboards for entire matches
	•	API mode for programmatic use

⸻

🤝 Contributing

Contributions are welcome!

Feel free to submit:
	•	bug fixes
	•	feature additions
	•	visualization upgrades
	•	tactical insights
	•	documentation improvements

Open an Issue or Pull Request on the repository.

⸻



⸻

📣 Acknowledgements
	•	SkillCorner for their OpenData initiative
	•	Open-source football analytics community
	•	Everyone contributing to accessible data-driven football analysis

⸻

