import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, Circle, Line, String, Group

BACKEND_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BACKEND_ROOT / "output"
DATA_DIR = BACKEND_ROOT / "data"

class PDFService:
    @staticmethod
    def generate_report_pdf(job_id: str, match_name: str) -> io.BytesIO:
        """
        Generates a 7-page professional PDF Tactical Coaching Report
        using actual generated match telemetry.
        """
        # Load associated files
        report_data = PDFService._load_json(OUTPUT_DIR / f"{job_id}_report_enriched.json") or PDFService._load_json(OUTPUT_DIR / f"{job_id}_report.json") or {}
        events_data = PDFService._load_json(OUTPUT_DIR / f"{job_id}_events.json") or {}
        tracking_data = PDFService._load_json(OUTPUT_DIR / f"{job_id}_tracking_data.json") or {}
        
        # Resolve metrics
        advice_items = report_data.get("advice_items") if isinstance(report_data, dict) else report_data or []
        summary_card = next((item for item in advice_items if item.get("flaw") == "Match Summary"), {})
        metrics = summary_card.get("summary_data") or PDFService._parse_evidence_metrics(summary_card.get("evidence", ""))
        
        # Initialize document
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=45,
            leftMargin=45,
            topMargin=54,
            bottomMargin=54
        )
        
        styles = getSampleStyleSheet()
        
        # Base colors
        c_forest = colors.HexColor("#0f3d1b")
        c_emerald = colors.HexColor("#10b981")
        c_dark_gray = colors.HexColor("#1f2937")
        c_light_gray = colors.HexColor("#f3f4f6")
        c_border = colors.HexColor("#e5e7eb")
        
        # Custom styles
        style_cover_title = ParagraphStyle(
            "CoverTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=32,
            leading=38,
            textColor=c_forest,
            spaceAfter=15
        )
        style_cover_subtitle = ParagraphStyle(
            "CoverSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=40
        )
        style_h1 = ParagraphStyle(
            "H1Heading",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=c_forest,
            spaceAfter=15,
            keepWithNext=True
        )
        style_h2 = ParagraphStyle(
            "H2Heading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=c_dark_gray,
            spaceAfter=8,
            keepWithNext=True
        )
        style_body = ParagraphStyle(
            "BodyTextDark",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=c_dark_gray,
            spaceAfter=12
        )
        style_quote = ParagraphStyle(
            "PhilosophyQuote",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#374151")
        )
        
        story = []
        
        # ---------------------------------------------------------
        # PAGE 1: COVER PAGE
        # ---------------------------------------------------------
        story.append(Spacer(1, 40))
        # Large color accent block
        d_accent = Drawing(520, 20)
        d_accent.add(Rect(0, 0, 520, 16, fillColor=c_forest, strokeColor=None))
        story.append(d_accent)
        story.append(Spacer(1, 30))
        
        story.append(Paragraph("GAFFER'S GUIDE", style_cover_subtitle))
        story.append(Paragraph("TACTICAL MATCH REPORT", style_cover_title))
        story.append(Paragraph(f"Analysis summary for the game: <b>{match_name}</b>", style_cover_subtitle))
        
        story.append(Spacer(1, 80))
        
        # Metadata Block Table
        meta_data = [
            [Paragraph("<b>Job Identifier:</b>", style_body), Paragraph(job_id, style_body)],
            [Paragraph("<b>Analysis Date:</b>", style_body), Paragraph(time.strftime("%Y-%m-%d %H:%M:%S UTC"), style_body)],
            [Paragraph("<b>Performance Profile:</b>", style_body), Paragraph(report_data.get("quality_profile", "balanced").upper(), style_body)],
            [Paragraph("<b>Chunking Interval:</b>", style_body), Paragraph(report_data.get("metadata", {}).get("chunking_interval", "15-minute intervals"), style_body)]
        ]
        t_meta = Table(meta_data, colWidths=[150, 350])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), c_light_gray),
            ('BOX', (0, 0), (-1, -1), 1, c_border),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8)
        ]))
        story.append(t_meta)
        
        story.append(PageBreak())
        
        # ---------------------------------------------------------
        # PAGE 2: EXECUTIVE SUMMARY
        # ---------------------------------------------------------
        story.append(Paragraph("Executive Summary", style_h1))
        summary_text = summary_card.get("evidence", "No executive summary evidence available.")
        story.append(Paragraph(summary_text, style_body))
        
        story.append(Spacer(1, 20))
        
        if advice_items:
            story.append(Paragraph("Primary Tactical Flaws Detected", style_h2))
            flaws_list = []
            for item in advice_items:
                if item.get("flaw") != "Match Summary":
                    flaws_list.append(f"• <b>{item.get('flaw')}</b> ({item.get('severity')} - {item.get('team', '').replace('_', ' ').title()}): {item.get('evidence')}")
            
            if flaws_list:
                for f_item in flaws_list[:5]: # top 5
                    story.append(Paragraph(f_item, style_body))
            else:
                story.append(Paragraph("No critical tactical flaws were flag-triggered in this session.", style_body))
                
        story.append(PageBreak())
        
        # ---------------------------------------------------------
        # PAGE 3: TACTICAL METRICS
        # ---------------------------------------------------------
        story.append(Paragraph("Tactical Performance Metrics", style_h1))
        story.append(Paragraph("Below is a side-by-side performance scorecard comparing Red Team (team_0) and Blue Team (team_1) average KPIs computed across all ingested video tracking frames.", style_body))
        story.append(Spacer(1, 15))
        
        t0 = metrics.get("team_0") or {}
        t1 = metrics.get("team_1") or {}
        
        metrics_table_data = [
            [Paragraph("<b>Performance KPI Category</b>", style_body), 
             Paragraph("<b>Red Team (Team 0)</b>", style_body), 
             Paragraph("<b>Blue Team (Team 1)</b>", style_body)],
            [Paragraph("Tactical Power Index (TPI)", style_body), f"{t0.get('tactical_power', 0.0):.1f}", f"{t1.get('tactical_power', 0.0):.1f}"],
            [Paragraph("Overall Compactness Score", style_body), f"{t0.get('compactness', 0.0):.1f}%", f"{t1.get('compactness', 0.0):.1f}%"],
            [Paragraph("Vertical Compactness", style_body), f"{t0.get('compactness_vertical', 0.0):.1f}%", f"{t1.get('compactness_vertical', 0.0):.1f}%"],
            [Paragraph("Horizontal Compactness", style_body), f"{t0.get('compactness_horizontal', 0.0):.1f}%", f"{t1.get('compactness_horizontal', 0.0):.1f}%"],
            [Paragraph("Midfield Block Compactness", style_body), f"{t0.get('compactness_midfield', 0.0):.1f}%", f"{t1.get('compactness_midfield', 0.0):.1f}%"],
            [Paragraph("Midfield Control Ratio", style_body), f"{t0.get('control', 50.0):.1f}%", f"{t1.get('control', 50.0):.1f}%"],
            [Paragraph("Pressing Intensity Score", style_body), f"{t0.get('intensity', 0.0):.1f}%", f"{t1.get('intensity', 0.0):.1f}%"],
            [Paragraph("Defensive Shape solidity", style_body), f"{t0.get('defensive_shape', 0.0):.1f}%", f"{t1.get('defensive_shape', 0.0):.1f}%"],
            [Paragraph("Transition Velocity Index", style_body), f"{t0.get('transition_speed', 0.0):.1f}", f"{t1.get('transition_speed', 0.0):.1f}"],
            [Paragraph("Win Probability Factor", style_body), f"{t0.get('win_prob', 50.0):.1f}%", f"{t1.get('win_prob', 50.0):.1f}%"]
        ]
        
        t_metrics = Table(metrics_table_data, colWidths=[220, 150, 150])
        t_metrics.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_forest),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_gray])
        ]))
        
        # Override headers text styling
        for col_idx in range(3):
            metrics_table_data[0][col_idx].style.textColor = colors.white
            
        story.append(t_metrics)
        story.append(PageBreak())
        
        # ---------------------------------------------------------
        # PAGE 4: TACTICAL RADAR
        # ---------------------------------------------------------
        story.append(Paragraph("Tactical Radar Spatial Map", style_h1))
        story.append(Paragraph("This map plots the average positions of both teams' players derived from the homography pitch coordinates, demonstrating overall defensive and attacking shapes.", style_body))
        story.append(Spacer(1, 20))
        
        # Draw the pitch vector graphic
        # Pitch width=380, height=240, centered around (190, 120)
        dw = Drawing(520, 300)
        
        # Green Pitch Grass
        dw.add(Rect(0, 0, 520, 300, fillColor=colors.HexColor("#e8f5e9"), strokeColor=c_forest, strokeWidth=2))
        
        # Boundary line
        px, py, pw, ph = 40, 30, 440, 240
        dw.add(Rect(px, py, pw, ph, fillColor=None, strokeColor=colors.white, strokeWidth=1.5))
        
        # Halfway Line
        dw.add(Line(px + pw/2, py, px + pw/2, py + ph, strokeColor=colors.white, strokeWidth=1.5))
        
        # Center Circle
        dw.add(Circle(px + pw/2, py + ph/2, 35, fillColor=None, strokeColor=colors.white, strokeWidth=1.5))
        dw.add(Circle(px + pw/2, py + ph/2, 2, fillColor=colors.white, strokeColor=colors.white))
        
        # Penalty Boxes
        dw.add(Rect(px, py + ph/2 - 60, 60, 120, fillColor=None, strokeColor=colors.white, strokeWidth=1.5))
        dw.add(Rect(px + pw - 60, py + ph/2 - 60, 60, 120, fillColor=None, strokeColor=colors.white, strokeWidth=1.5))
        
        # Goal Areas
        dw.add(Rect(px, py + ph/2 - 25, 20, 50, fillColor=None, strokeColor=colors.white, strokeWidth=1.2))
        dw.add(Rect(px + pw - 20, py + ph/2 - 25, 20, 50, fillColor=None, strokeColor=colors.white, strokeWidth=1.2))
        
        # Plot player centroids from tracking data
        frames_list = tracking_data.get("frames", [])
        
        # Calculate overall centroids
        player_coords: Dict[str, Dict[str, List[float]]] = {}
        for f in frames_list:
            for p in f.get("players", []):
                pid = str(p.get("player_id"))
                team = p.get("team") # "team_0" or "team_1"
                coords = p.get("radar_pt") # [x, y] in meters (-52.5 to 52.5, -34.0 to 34.0)
                
                if coords and len(coords) == 2 and not any(math.isnan(v) for v in coords):
                    if pid not in player_coords:
                        player_coords[pid] = {"team": team, "x": [], "y": []}
                    player_coords[pid]["x"].append(coords[0])
                    player_coords[pid]["y"].append(coords[1])
                    
        # Average them
        t0_dots = []
        t1_dots = []
        for pid, d in player_coords.items():
            if d["x"] and d["y"]:
                avg_x = sum(d["x"]) / len(d["x"])
                avg_y = sum(d["y"]) / len(d["y"])
                
                # Pitch coordinates are X: -52.5 to 52.5, Y: -34 to 34
                # We map this to px to px+pw (40 to 480) and py to py+ph (30 to 270)
                screen_x = px + pw/2 + (avg_x / 52.5) * (pw / 2)
                screen_y = py + ph/2 - (avg_y / 34.0) * (ph / 2) # invert Y
                
                # Boundary safety check
                screen_x = max(px + 5, min(screen_x, px + pw - 5))
                screen_y = max(py + 5, min(screen_y, py + ph - 5))
                
                if d["team"] == "team_0":
                    t0_dots.append((screen_x, screen_y, pid))
                else:
                    t1_dots.append((screen_x, screen_y, pid))
                    
        # Plot Red dots (Team 0)
        for x, y, pid in t0_dots:
            dw.add(Circle(x, y, 6, fillColor=colors.HexColor("#ef4444"), strokeColor=colors.white, strokeWidth=0.8))
            dw.add(String(x - 3.5, y - 3, pid, fontSize=6.5, fontName="Helvetica-Bold", fillColor=colors.white))
            
        # Plot Blue dots (Team 1)
        for x, y, pid in t1_dots:
            dw.add(Circle(x, y, 6, fillColor=colors.HexColor("#3b82f6"), strokeColor=colors.white, strokeWidth=0.8))
            dw.add(String(x - 3.5, y - 3, pid, fontSize=6.5, fontName="Helvetica-Bold", fillColor=colors.white))
            
        # Plot centroid legend
        dw.add(Rect(px + 10, py + 10, 110, 35, fillColor=colors.HexColor("#0a0f0a"), strokeColor=colors.white, strokeWidth=0.5))
        dw.add(Circle(px + 20, py + 32, 4, fillColor=colors.HexColor("#ef4444"), strokeColor=colors.white, strokeWidth=0.5))
        dw.add(String(px + 30, py + 29, "Red Team (0)", fontSize=7, fontName="Helvetica", fillColor=colors.white))
        dw.add(Circle(px + 20, py + 18, 4, fillColor=colors.HexColor("#3b82f6"), strokeColor=colors.white, strokeWidth=0.5))
        dw.add(String(px + 30, py + 15, "Blue Team (1)", fontSize=7, fontName="Helvetica", fillColor=colors.white))
        
        story.append(dw)
        story.append(Spacer(1, 15))
        story.append(Paragraph("<i>Note: Player coordinates are averaged across their active tracking duration. Labels correspond to the tracked player IDs.</i>", style_quote))
        
        story.append(PageBreak())
        
        # ---------------------------------------------------------
        # PAGE 5: MATCH EVENTS
        # ---------------------------------------------------------
        story.append(Paragraph("Key Match Events", style_h1))
        story.append(Paragraph("This table details key tactical events (Sprints, entries into critical zones, turnovers) processed by Gaffer's Event Ontology catalog.", style_body))
        story.append(Spacer(1, 15))
        
        raw_events = events_data.get("events", [])
        # Take top 15 events sorted by importance
        raw_events.sort(key=lambda x: x.get("importance", 0.0), reverse=True)
        top_events = raw_events[:15]
        
        events_table_data = [
            [Paragraph("<b>Time</b>", style_body), 
             Paragraph("<b>Event Type</b>", style_body), 
             Paragraph("<b>Player</b>", style_body), 
             Paragraph("<b>Pitch Zone</b>", style_body), 
             Paragraph("<b>Description</b>", style_body)]
        ]
        
        for ev in top_events:
            t_sec = ev.get("start_time_s", 0.0)
            t_str = f"{int(t_sec // 60):02d}:{int(t_sec % 60):02d}"
            pid = f"P{ev.get('player_id')}" if ev.get('player_id') is not None else "—"
            zone = ev.get("pitch_zone", "—").replace("_", " ").title()
            desc = ev.get("description", "")
            
            events_table_data.append([
                Paragraph(t_str, style_body),
                Paragraph(ev.get("event_name", "Event"), style_body),
                Paragraph(pid, style_body),
                Paragraph(zone, style_body),
                Paragraph(desc, style_body)
            ])
            
        t_events = Table(events_table_data, colWidths=[45, 95, 45, 95, 240])
        t_events.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_forest),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_gray])
        ]))
        
        # Override headers text styling
        for col_idx in range(5):
            events_table_data[0][col_idx].style.textColor = colors.white
            
        story.append(t_events)
        story.append(PageBreak())
        
        # ---------------------------------------------------------
        # PAGE 6: PLAYER PHYSICAL & TACTICAL ANALYSIS
        # ---------------------------------------------------------
        story.append(Paragraph("Player Performance Registry", style_h1))
        story.append(Paragraph("Calculated physical outputs and threat ranking scores aggregated across all tracked segments of the video.", style_body))
        story.append(Spacer(1, 15))
        
        # Calculate Player metrics dynamically from tracking data
        players_registry: Dict[str, Dict[str, Any]] = {}
        for f in frames_list:
            for p in f.get("players", []):
                pid = str(p.get("player_id"))
                team = p.get("team", "team_0")
                coords = p.get("radar_pt")
                speed = p.get("speed_kmh", 0.0) or 0.0
                
                if pid not in players_registry:
                    players_registry[pid] = {
                        "id": pid,
                        "team": team,
                        "distance": 0.0,
                        "speeds": [],
                        "sprints": 0,
                        "in_sprint": False,
                        "last_pt": None
                    }
                
                # Distance accumulation
                reg = players_registry[pid]
                if reg["last_pt"] and coords:
                    dx = coords[0] - reg["last_pt"][0]
                    dy = coords[1] - reg["last_pt"][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 15.0: # ignore noise jumps
                        reg["distance"] += dist
                
                reg["last_pt"] = coords
                reg["speeds"].append(speed)
                
                # Sprint count (hysteresis)
                if speed > 24.0 and not reg["in_sprint"]:
                    reg["in_sprint"] = True
                    reg["sprints"] += 1
                elif speed < 19.0 and reg["in_sprint"]:
                    reg["in_sprint"] = False

        # Load Threat Scores
        threats_path = OUTPUT_DIR / f"{job_id}_threat_profiles.json"
        threat_scores = {}
        if threats_path.is_file():
            try:
                with open(threats_path, "r") as f_th:
                    t_list = json.load(f_th)
                    for item in t_list:
                        threat_scores[str(item["player_id"])] = item.get("threat_score", 0.0)
            except Exception:
                pass

        player_table_data = [
            [Paragraph("<b>Player ID</b>", style_body), 
             Paragraph("<b>Team</b>", style_body), 
             Paragraph("<b>Distance (m)</b>", style_body), 
             Paragraph("<b>Avg Speed (km/h)</b>", style_body), 
             Paragraph("<b>Max Speed (km/h)</b>", style_body), 
             Paragraph("<b>Sprints</b>", style_body), 
             Paragraph("<b>Threat Score</b>", style_body)]
        ]
        
        # Sort by Player ID numerically
        sorted_player_keys = sorted(players_registry.keys(), key=lambda x: int(x) if x.isdigit() else 999)
        
        for p_key in sorted_player_keys:
            p_data = players_registry[p_key]
            d_covered = p_data["distance"]
            speeds = p_data["speeds"]
            avg_sp = (sum(speeds) / len(speeds)) if speeds else 0.0
            max_sp = max(speeds) if speeds else 0.0
            t_score = threat_scores.get(p_key, 0.0)
            team_lbl = "Red" if p_data["team"] == "team_0" else "Blue"
            
            player_table_data.append([
                f"P{p_key}",
                team_lbl,
                f"{d_covered:.1f}",
                f"{avg_sp:.1f}",
                f"{max_sp:.1f}",
                str(p_data["sprints"]),
                f"{t_score:.1f}/100"
            ])
            
        t_players = Table(player_table_data, colWidths=[65, 65, 85, 95, 95, 55, 60])
        t_players.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_forest),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_gray])
        ]))
        
        # Override headers text styling
        for col_idx in range(7):
            player_table_data[0][col_idx].style.textColor = colors.white
            
        story.append(t_players)
        story.append(PageBreak())
        
        # ---------------------------------------------------------
        # PAGE 7: AI COACHING RECOMMENDATIONS
        # ---------------------------------------------------------
        story.append(Paragraph("AI Tactical Recommendations", style_h1))
        story.append(Paragraph("Specific tactical tweaks generated by Gaffer's RAG rules engine and augmented with large language model guidance.", style_body))
        story.append(Spacer(1, 15))
        
        recommendations = [item for item in advice_items if item.get("flaw") != "Match Summary"]
        
        if not recommendations:
            story.append(Paragraph("No specific coaching adjustments suggested for this session.", style_body))
        else:
            for item in recommendations[:4]: # top 4 adjustments to fit on one page
                rec_story = []
                severity = item.get("severity", "Info")
                team_lbl = "Red Team (0)" if item.get("team") == "team_0" else "Blue Team (1)"
                
                # Title block
                rec_story.append(Paragraph(
                    f"<b>{item.get('flaw')}</b> &mdash; <i>{team_lbl}</i>", style_h2
                ))
                
                # Severity Badge
                sev_color = "#3b82f6"
                if severity == "Critical":
                    sev_color = "#ef4444"
                elif severity == "High":
                    sev_color = "#f97316"
                elif severity == "Medium":
                    sev_color = "#eab308"
                rec_story.append(Paragraph(
                    f"<font color='{sev_color}'><b>[{severity.upper()} SEVERITY]</b></font>", style_body
                ))
                
                # Evidence
                rec_story.append(Paragraph(f"<b>Evidence:</b> {item.get('evidence', '')}", style_body))
                
                # Instruction
                instr = item.get("tactical_instruction")
                if instr:
                    rec_story.append(Paragraph(f"<b>Coaching Instruction:</b> {instr}", style_body))
                    
                # Divider line
                rec_story.append(Spacer(1, 5))
                story.append(KeepTogether(rec_story))
                story.append(Spacer(1, 15))
                
        # Build Document with Footer Page Template
        def _add_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.HexColor("#6b7280"))
            # Left Footer
            canvas.drawString(45, 30, "Gaffer's Guide — Match Intelligence Report")
            # Right Footer
            canvas.drawRightString(doc.pagesize[0] - 45, 30, f"Page {doc.page}")
            canvas.restoreState()
            
        doc.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)
        buffer.seek(0)
        return buffer

    @staticmethod
    def _load_json(path: Path) -> Optional[Any]:
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _parse_evidence_metrics(evidence: str) -> Dict[str, Any]:
        """Fallbacks parsing of metrics from match summary string."""
        import re
        parsed = {"team_0": {}, "team_1": {}, "win_probability": {}}
        tp = re.search(r"Tactical Power:\s*Red\s*([\d\.]+)\s*vs\s*Blue\s*([\d\.]+)", evidence)
        if tp:
            parsed["team_0"]["tactical_power"] = float(tp.group(1))
            parsed["team_1"]["tactical_power"] = float(tp.group(2))
            parsed["team_red_score"] = float(tp.group(1))
            parsed["team_blue_score"] = float(tp.group(2))
            
        wp = re.search(r"Win Probability:\s*Red\s*([\d\.]+)%[^|]*\|\s*Blue\s*([\d\.]+)%", evidence)
        if wp:
            parsed["win_probability"]["team_red"] = float(wp.group(1))
            parsed["win_probability"]["team_blue"] = float(wp.group(2))
            parsed["team_0"]["win_prob"] = float(wp.group(1))
            parsed["team_1"]["win_prob"] = float(wp.group(2))
            
        cp = re.search(r"Compactness:\s*Red\s*([\d\.]+)\s*/\s*Blue\s*([\d\.]+)", evidence)
        if cp:
            parsed["team_0"]["compactness"] = float(cp.group(1))
            parsed["team_1"]["compactness"] = float(cp.group(2))
            
        ts = re.search(r"Transition Speed:\s*Red\s*([\d\.]+)\s*/\s*Blue\s*([\d\.]+)", evidence)
        if ts:
            parsed["team_0"]["transition_speed"] = float(ts.group(1))
            parsed["team_1"]["transition_speed"] = float(ts.group(2))
            
        return parsed
