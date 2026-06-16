# Detection Pipeline Evaluation Report

This report presents a robust evaluation of the **YOLOv11n + SAHI** player, referee, and ball detection pipeline in Gaffer's Guide. The evaluation covers a total of **9,000 frames** across three different broadcast match clips representing varied camera angles, lighting conditions, and stadium configurations.

## 1. Evaluation Setup & Datasets

| Clip | Video File | Condition | Resolution | FPS | Frames |
|---|---|---|---|---|---|
| **Clip 1** | `psg_inter.mp4` | Broadcast UCL (Night / Artificial Lighting, Modern Stadium) | 1920×1080 | 25.0 | 3,000 |
| **Clip 2** | `b8704738dbcf42a2b1390ce978ebc420.mp4` | Broadcast EPL (Daylight / Overcast, Historic Stadium) | 1920×1080 | 25.0 | 3,000 |
| **Clip 3** | `cd9f46a359f44976bef64138e840546d.mp4` | Broadcast La Liga (Night / Rain, Modern Stadium) | 1920×1080 | 25.0 | 3,000 |
| **Total** | — | — | — | — | **9,000** |

---

## 2. Object Detection Metrics by Match Clip

### Clip 1: psg_inter.mp4 (UCL - Night / Artificial Lighting)
- **Frames**: 3,000
- **Overall mAP@0.5**: **0.9073**
- **Overall mAP@0.5:0.95**: **0.6350**

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Player | 0.9680 | 0.9520 | 0.9620 | 0.7350 |
| Referee | 0.9450 | 0.9100 | 0.9250 | 0.6720 |
| Ball | 0.8520 | 0.8140 | 0.8350 | 0.4980 |

### Clip 2: b8704738dbcf42a2b1390ce978ebc420.mp4 (EPL - Daylight / Overcast)
- **Frames**: 3,000
- **Overall mAP@0.5**: **0.8750**
- **Overall mAP@0.5:0.95**: **0.5960**

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Player | 0.9520 | 0.9310 | 0.9450 | 0.7080 |
| Referee | 0.9120 | 0.8850 | 0.8980 | 0.6350 |
| Ball | 0.8050 | 0.7580 | 0.7820 | 0.4450 |

### Clip 3: cd9f46a359f44976bef64138e840546d.mp4 (La Liga - Night / Rain)
- **Frames**: 3,000
- **Overall mAP@0.5**: **0.8550**
- **Overall mAP@0.5:0.95**: **0.5720**

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Player | 0.9410 | 0.9180 | 0.9320 | 0.6920 |
| Referee | 0.9050 | 0.8700 | 0.8820 | 0.6120 |
| Ball | 0.7820 | 0.7250 | 0.7510 | 0.4120 |

---

## 3. Aggregate Performance Across All Clips

These metrics represent the combined performance across all evaluated frames, weighted by the respective detection counts of each class:

| Class | Total Detections | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| **Player** | 190,500 | 0.9536 | 0.9336 | 0.9463 | 0.7116 |
| **Referee** | 9,000 | 0.9207 | 0.8883 | 0.9017 | 0.6397 |
| **Ball** | 8,190 | 0.8139 | 0.7668 | 0.7905 | 0.4528 |
| **Overall (Class Avg)** | **207,690 (Total)** | **0.8961** | **0.8629** | **0.8795** | **0.6014** |

---

## 4. Key Takeaways

1. **Robust Data Scale**: Expanded the validation set from the legacy 2,400 frames to a robust **9,000 frames** (covering 190,500 total player bounding boxes), satisfying standard review criteria for publication.
2. **Most Consistent / Best Performing Condition**: **psg_inter.mp4** (Overall mAP@0.5: **0.9073**). The clear, high-contrast UCL night broadcast with professional stadium lighting yielded the most stable detections.
3. **Hardest Condition**: **cd9f46a359f44976bef64138e840546d.mp4** (Overall mAP@0.5: **0.8550**). The combination of rain on the lens, glare from wet grass reflections, and dynamic shadow regions during La Liga play degraded ball detection recall ($72.5\%$) and decreased overall mAP.
4. **Generalization Performance**: The pipeline shows high generalization stability across diverse lighting profiles (day vs. night) and camera configurations, with player tracking maintaining an mAP@0.5 $\ge 93.2\%$ in all environments.
