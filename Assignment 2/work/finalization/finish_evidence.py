from pathlib import Path
import re,json,csv
H=Path(__file__).resolve().parent;P=H.parents[1]/'outputs/01a1af69-d8fe-71f1-a2cd-78094fad2cb8/Submission_Package'
f=P/'report/main.tex';t=f.read_text()
start=t.index('\\begin{table}[H]',t.index('\\vspace{-6pt}'))
end=t.index('\\end{table}',start)+len('\\end{table}')
t=t[:start]+r'''\textbf{RQ1 (Shahoud).} How do studies measure the stability of SHAP and LIME explanations for tabular models, and under which perturbation or resampling regimes? Key terms: stability metric, perturbation regime, prediction-preserving and noise floor. This question identifies defensible metrics and a perturbation protocol.

\textbf{RQ2 (Saif).} What evidence shows whether stable or accurate predictions indicate stable SHAP and LIME explanations, and when does that link break? Key terms: prediction change, explanation drift and decoupling. This tests the assumption linking prediction reliability to explanation reliability.

\textbf{RQ3 (joint).} Which mechanisms (split thresholds, bin edges, decision boundaries, collinearity, sampling) explain SHAP and LIME instability on tabular models, and can instability be predicted without running perturbations? Key terms: split threshold, bin edge, decision boundary, fragility score and uncertainty score. This explains why explanations change and assesses possible warning scores.
'''+t[end:]
t=t.replace('a note explains every score below 1.', 'the Quality sheet summarizes concerns, and the supplied item-evidence table links every score to extracted evidence and its source locator.')
t=t.replace('workbook sheets \\texttt{Extraction}, \\texttt{Quality}', 'workbook sheets \\texttt{Extraction}, \\texttt{Quality} and \\texttt{Quality\\_Evidence}')
f.write_text(t)
print('Review questions compacted without changing their scope.')
