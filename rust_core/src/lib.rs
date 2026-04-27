use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
struct TemporalSearchRegion {
    #[pyo3(get)]
    xyxy: (i32, i32, i32, i32),
    #[pyo3(get)]
    radius_px: i32,
}

#[pyclass]
struct TemporalBallPrior {
    base_radius: i32,
    max_radius: i32,
    expand_step: i32,
    confidence_reset: f64,
    last_ball_xy: Option<(f64, f64)>,
    miss_streak: i32,
}

#[pymethods]
impl TemporalBallPrior {
    #[new]
    #[pyo3(signature = (*, base_radius_px=160, max_radius_px=520, miss_expand_step_px=40, confidence_reset=0.55))]
    fn new(
        base_radius_px: i32,
        max_radius_px: i32,
        miss_expand_step_px: i32,
        confidence_reset: f64,
    ) -> Self {
        let base_radius = base_radius_px.max(16);
        let max_radius = max_radius_px.max(base_radius);
        Self {
            base_radius,
            max_radius,
            expand_step: miss_expand_step_px.max(1),
            confidence_reset,
            last_ball_xy: None,
            miss_streak: 0,
        }
    }

    #[getter]
    fn last_ball_xy(&self) -> Option<(f64, f64)> {
        self.last_ball_xy
    }

    fn on_detection(&mut self, center_xy: (f64, f64), confidence: f64) {
        self.last_ball_xy = Some(center_xy);
        if confidence >= self.confidence_reset {
            self.miss_streak = 0;
        }
    }

    fn on_miss(&mut self) {
        self.miss_streak += 1;
    }

    fn current_radius_px(&self) -> i32 {
        let expanded = self.base_radius + self.expand_step * self.miss_streak;
        expanded.min(self.max_radius)
    }

    fn search_region(&self, frame_w: i32, frame_h: i32) -> Option<TemporalSearchRegion> {
        let (cx, cy) = self.last_ball_xy?;
        let radius = self.current_radius_px();
        let x1 = (cx as i32 - radius).max(0);
        let y1 = (cy as i32 - radius).max(0);
        let x2 = (cx as i32 + radius).min(frame_w);
        let y2 = (cy as i32 + radius).min(frame_h);
        if x2 <= x1 || y2 <= y1 {
            return None;
        }
        Some(TemporalSearchRegion {
            xyxy: (x1, y1, x2, y2),
            radius_px: radius,
        })
    }
}

#[pyfunction]
fn rank_candidates_rs(
    candidates: Vec<(f64, f64, f64, f64, f64)>,
    temporal_anchor_xy: Option<(f64, f64)>,
    search_radius_px: i32,
) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    if temporal_anchor_xy.is_none() || search_radius_px <= 0 {
        return candidates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.4.total_cmp(&b.4))
            .map(|(idx, _)| idx);
    }

    let (ax, ay) = temporal_anchor_xy.unwrap();
    let radius = (search_radius_px.max(1)) as f64;
    candidates
        .iter()
        .enumerate()
        .map(|(idx, (x1, y1, x2, y2, confidence))| {
            let cx = (x1 + x2) * 0.5;
            let cy = (y1 + y2) * 0.5;
            let dx = cx - ax;
            let dy = cy - ay;
            let dist = (dx * dx + dy * dy).sqrt();
            let proximity = (1.0 - (dist / radius)).max(0.0);
            let score = confidence + 0.35 * proximity;
            (idx, score)
        })
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
}

#[pymodule]
fn gaffers_core_math(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TemporalSearchRegion>()?;
    m.add_class::<TemporalBallPrior>()?;
    m.add_function(wrap_pyfunction!(rank_candidates_rs, m)?)?;
    Ok(())
}
