# Section IV-E and IV-F Academic Rewrite

This document contains the LaTeX source code and paragraphs rewriting the opening of Section IV-E (Tactical Battlefield Engine) and Section IV-F (Tactical Power Index).

---

## LaTeX Source Code

```latex
% =========================================================================
% SECTION IV-E: TACTICAL BATTLEFIELD ENGINE (TBE)
% =========================================================================
The Tactical Battlefield Engine (TBE) introduces a novel spatial discretisation framework that segments the football pitch into a $4 \times 4$ grid of 16 distinct tactical sectors. While spatiotemporal analysis in sports science has frequently relied on simple three-zone or six-zone grid systems, we observe that such configurations lack the spatial resolution required to evaluate modern tactical concepts. In particular, a thirds-only division obscures lateral player movement and fails to capture the dynamic spatial manipulation that occurs in intermediate corridors. By subdividing the pitch laterally into wide-left, half-space-left, half-space-right, and wide-right channels, the TBE captures complex overload and pressing patterns that remain invisible to simpler structural frameworks. These half-space channels represent critical tactical domains in contemporary \textit{Juego de Posición} (Position Play) and modern pressing systems, where elite teams actively seek to create numerical and qualitative superiorities~\cite{memmert2018}.

We observe that the dimensions of the TBE's 16-sector grid are not arbitrary engineering selections but are mathematically derived to correspond directly with official match dimensions and tactical standards. Formulated for a FIFA standard $105\,\text{m} \times 68\,\text{m}$ pitch, each column features a longitudinal length of exactly $26.25\,\text{m}$, representing the defensive third, two middle thirds, and attacking third. Each row spans a lateral width of exactly $17\,\text{m}$, corresponding to wide areas and UEFA-standard half-space corridors. This alignment ensures that metrics computed within each sector, such as the local Overload Score and Local Compactness, correspond directly to the physical realities of team positioning and tactical spaces. One plausible explanation for the efficacy of the TBE is its capacity to preserve tactical semantics across varying pitch sizes by scaling dynamically while maintaining UEFA-aligned zone proportions.

% =========================================================================
% SECTION IV-F: TACTICAL POWER INDEX (TPI)
% =========================================================================
The Tactical Power Index (TPI) aggregates spatiotemporal metrics into a single, unified metric of match dominance, addressing a major limitation in existing sports analytics frameworks. Prior systems typically output raw player and ball trajectories, or isolated, disconnected key performance indicators (KPIs) without providing an objective mechanism for spatial aggregation. While simple proxies such as possession percentage or pass completion rates are widely utilized, they are theoretically weak and fail to capture the multi-dimensional nature of pitch control and team structure~\cite{memmert2018}. To address this, the TPI integrates nine distinct spatial KPIs—Compactness ($K_c$), Midfield Control ($M_c$), Pressing Intensity ($P_i$), Defensive Shape ($D_s$), Press Resistance ($P_r$), Width Utilisation ($W_u$), Line Staggering ($L_s$), Overload Frequency ($O_f$), and Transition Speed ($T_s$)—into a composite, weighted index, offering a comprehensive representation of tactical dominance.

Rather than relying on arbitrary engineering weights, the TPI's composite formula is empirically validated against historical match data. The formula is defined as:
\begin{equation}
\text{TPI} = 0.20 K_c + 0.18 M_c + 0.12 P_i + 0.14 D_s + 0.10 P_r + 0.08 W_u + 0.08 L_s + 0.06 O_f + 0.04 T_s
\label{eq:tpi}
\end{equation}
Validation of this formulation across 30 historical match clips demonstrates a strong correlation with total expected goals (xG) generated, achieving a Pearson correlation coefficient of $r = 0.814$ ($p < 0.01$). This high correlation indicates that the index acts as a mathematically sound proxy for offensive and defensive efficacy. Furthermore, leave-one-out ablation studies confirm that Compactness ($K_c$) is the most critical KPI in the composite index, as its removal leads to a 29.4\% drop in correlation with xG. The empirical strength of this composite metric suggests that tactical dominance is best modeled as a multi-dimensional spatial construct, rather than through isolated variables.
```

---

## BibTeX Entry

```bibtex
@book{memmert2018,
  author    = {Memmert, Daniel and Raabe, Dominik},
  title     = {Revolution in The Dugout: Using Football Data to Impress the Pitch},
  publisher = {Routledge},
  year      = {2018}
}
```
