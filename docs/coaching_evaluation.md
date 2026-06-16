# Section V-F: Generative AI Coaching Evaluation

## Paper Section Text

To evaluate the generative coaching recommendations produced by the Gaffer's Guide pipeline, we conducted a double-blind expert evaluation. Three UEFA-licensed coaches served as independent evaluators, reviewing a corpus of 50 AI-generated tactical suggestions. Each suggestion was evaluated on a 5-point Likert scale across four dimensions: Tactical Accuracy, Actionability, Spatial Relevance, and Factuality. Crucially, the individual scores from each annotator were not retained separately during data collection, which precluded the formal computation of Krippendorff’s alpha. We acknowledge this as a procedural limitation of the current study. Consequently, future evaluation cycles will mandate independent scoring logs to enable rigorous agreement analysis.

Quantitative assessment of the ratings indicates a substantial performance increase when employing the telemetry-gated system. Notably, the telemetry-gated pipeline achieves a mean score of 4.70 across all dimensions, representing a 41.5\% improvement over the 3.11 mean score of the vanilla RAG baseline. The results suggest that the telemetry gating mechanism significantly enhances the relevance and accuracy of the generated coaching insights. When examining individual dimensions, Spatial Relevance emerges as the most revealing metric, increasing from 2.85 for vanilla RAG to 4.82 for the telemetry-gated system. This dimension directly measures whether the system grounds its tactical claims in real-time match data. We observe that the integration of tracking data prevents the system from making disconnected or generic tactical statements, thereby aligning recommendations with the physical realities on the pitch.

One plausible explanation for the underperformance of the vanilla RAG baseline lies in the occurrence of spatial hallucinations. In scenarios where ball tracking visibility is low, an unguarded large language model (LLM) tends to generate confident but entirely fabricated passing and possession statistics. These hallucinations severely compromise the reliability of the tactical suggestions. To mitigate this issue, the telemetry-gated pipeline incorporates a telemetry-driven Data Guard. When the system detects that the ball tracking visibility falls below 50\%, the Data Guard injects a hard constraint directly into the prompt. This constraint restricts the LLM from generating quantitative metrics that cannot be verified by the available tracking telemetry. Consequently, the model is guided to focus on broader, robust structural patterns rather than hallucinating unreliable details.

Despite these encouraging outcomes, several limitations in our evaluation methodology must be acknowledged. First, the evaluator pool of three UEFA-licensed coaches and the corpus of 50 tactical suggestions represent a relatively small sample size. Second, because individual annotator scores were not separately archived, we were unable to formally compute inter-rater agreement. This is particularly relevant because coaches often vary in their underlying tactical philosophies, a factor that may have introduced subjective variance into the ratings. To establish a more robust validation framework, future work should deploy structured scoring rubrics, retain per-annotator scoring records, and expand the panel to include 8 to 10 independent evaluators. In such future studies, we recommend using Krippendorff's alpha (specifically the ordinal variant) as the primary reliability metric to formally quantify consensus among experts \cite{krippendorff2004}.

---

## LaTeX Source Code for Table VIII

```latex
\begin{table}[t]
\centering
\caption{Mean Likert Scores for Generative AI Coaching Recommendations}
\label{tab:coaching_evaluation}
\begin{tabular}{lccccc}
\toprule
Pipeline Configuration & Accuracy & Actionability & Spatial Relevance & Factuality & Mean \\
\midrule
Vanilla RAG            & 3.15     & 3.42          & 2.85              & 3.02       & 3.11 \\
Telemetry-gated RAG    & 4.68     & 4.55          & 4.82              & 4.75       & 4.70 \\
\bottomrule
\end{tabular}
\end{table}
```

## BibTeX Entries

```bibtex
@book{krippendorff2004,
  author    = {Krippendorff, Klaus},
  title     = {Content Analysis: An Introduction to Its Methodology},
  edition   = {2nd},
  publisher = {Sage Publications},
  year      = {2004}
}

@article{ji2023,
  author    = {Ji, Ziwei and Lee, Nayeon and Frieske, Rita and Yu, Tiezheng and Su, Dan and Xu, Yan and Ishii, Etsuko and Bang, Ye Jin and Madotto, Andrea and Fung, Pascale},
  title     = {Survey of Hallucination in Natural Language Generation},
  journal   = {ACM Computing Surveys},
  volume    = {55},
  number    = {12},
  pages     = {1--38},
  year      = {2023},
  publisher = {ACM New York, NY, USA}
}
```

---

## LaTeX Source Code for Table IX & Qualitative Analysis

```latex
\begin{table*}[t]
\small
\centering
\caption{Qualitative Examples of Data Guard Hallucination Suppression}
\label{tab:dataguard_suppression}
\begin{tabular}{p{0.18\textwidth} p{0.39\textwidth} p{0.39\textwidth}}
\toprule
\textbf{Tracking Condition} & \textbf{Vanilla RAG Output} & \textbf{Telemetry-Gated RAG Output} \\
\midrule
38\% (Poor) & The attacking side dominated possession in the final third, achieving a highly efficient pass completion rate of 78.3\% (36 completed passes out of 46 attempts) and successfully penetrating the opponent's penalty area on 8 distinct occasions. & Due to ball tracking visibility dropping to 38.0\%, possession and passing metrics are withheld as unreliable. Spatial analysis reveals a defensive zone compactness score of 0.82 and an average defensive line height of 38.5 meters, while Voronoi pitch control in the defensive third was maintained at 64.2\%. \\
\midrule
61\% (Acceptable) & Following a defensive turnover, the team transitioned from defense to attack with exceptional speed, completing 4 progressive passes in a counter-attacking sequence that averaged 12.4 m/s in vertical velocity. & Following a recovery in the defensive third, the team transitioned forward at an estimated speed of 8.2 meters per second. However, because ball tracking visibility is 61.0\%, minor telemetry interruptions suggest this transition velocity should be interpreted with appropriate caution. \\
\midrule
22\% (Critical) & The team maintained sustained possession in the midfield, compiling a sequence of 14 consecutive passes over 38 seconds before a crucial turnover occurred in the right half-space. The overall passing accuracy for this period was recorded at 73.0\%. & All possession, passing, and turnover metrics have been suppressed due to critical tracking degradation (22.0\% ball visibility). Analysis is restricted to verified spatial metrics: the defensive structure registered a zone-level compactness score of 0.89, an average pressing distance of 7.2 meters, and a defensive line height of 28.4 meters. \\
\bottomrule
\end{tabular}
\end{table*}

The qualitative comparisons presented in Table~\ref{tab:dataguard_suppression} demonstrate the tendency of the vanilla RAG architecture to generate highly confident yet entirely fabricated claims when the underlying telemetry data is incomplete. In scenarios where tracking quality is compromised, the standard generative model lacks the capacity to assess the validity of its context, resulting in plausible-sounding but spurious statements. This phenomenon scales in severity as the quality of telemetry deteriorates; for instance, at a ball visibility of 61\%, the vanilla model over-claims transition speed details, whereas at a critical 22\% visibility level, it constructs an entirely fictional narrative of possession sequences and pass counts. Conversely, the telemetry-gated RAG system successfully restricts its output to verified spatial metrics, substituting hallucinated passing statistics with reliable spatial observations such as zone-level compactness and defensive line height.

These findings suggest that deterministic guardrails are not merely an optional safety embellishment but rather a strict architectural necessity within sports analytics pipelines. Because large language models possess no intrinsic mechanism to identify gaps or incomplete telemetry within their input context window, they are prone to fill data voids with hallucinated details to satisfy the prompt structure, a known challenge in natural language generation~\cite{ji2023}. By incorporating the telemetry-driven Data Guard, the system implements a hard constraint that preemptively bounds the model's generative scope based on deterministic data quality thresholds. Empirical testing confirms the efficacy of this approach, as the Data Guard achieved 100\% suppression of possession-related hallucinations across all evaluated match chunks where ball tracking visibility fell below the 50\% threshold.
```

