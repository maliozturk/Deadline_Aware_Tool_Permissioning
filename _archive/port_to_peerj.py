import re
from pathlib import Path

def main():
    tex_dir = Path("Tex")
    ms_path = tex_dir / "manuscript.tex"
    main_path = tex_dir / "PeerJ" / "main.tex"
    
    with open(ms_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Extract abstract
    abstract_match = re.search(r"\\begin{abstract}(.*?)\\end{abstract}", content, re.DOTALL)
    if not abstract_match:
        raise ValueError("Could not find abstract in manuscript.tex")
    abstract = abstract_match.group(1).strip()
    
    # Extract keywords
    keyword_match = re.search(r"\\begin{keyword}(.*?)\\end{keyword}", content, re.DOTALL)
    if not keyword_match:
        raise ValueError("Could not find keywords in manuscript.tex")
    keywords_raw = keyword_match.group(1).strip()
    keywords = keywords_raw.replace("\\sep", ",").strip()
    
    # Extract body
    # Starts at \section{Introduction} and ends before \bibliographystyle{elsarticle-num}
    body_start_idx = content.find("\\section{Introduction}")
    if body_start_idx == -1:
        raise ValueError("Could not find \\section{Introduction} in manuscript.tex")
        
    body_end_idx = content.find("\\bibliographystyle{elsarticle-num}")
    if body_end_idx == -1:
        # Fallback if style is different
        body_end_idx = content.find("\\bibliography{")
        
    if body_end_idx == -1:
        raise ValueError("Could not find bibliography boundary in manuscript.tex")
        
    body = content[body_start_idx:body_end_idx].strip()
    
    # Template for wlpeerj
    template = r"""%% Submissions for peer-review must enable line-numbering 
%% using the lineno option in the \documentclass command.
%%
%% Camera-ready submissions do not need line numbers, and
%% should have this option removed.
%%
%% Please note that the line numbering option requires
%% version 1.1 or newer of the wlpeerj.cls file, and
%% the corresponding author info requires v1.2

\documentclass[fleqn,10pt,lineno]{wlpeerj}

% --- Additional Packages and environments required by the manuscript ---
\usepackage{amsthm}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{multirow}
\usepackage{url}
\usepackage{mathtools}

\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}

\title{Context-Aware Dynamic Tool Resolution for Deadline-Constrained Autonomous Agents}

\author[1]{Anonymous}
\affil[1]{Blinded for Review}
\corrauthor[1]{Anonymous}{anonymous@example.com}

\keywords{""" + keywords + r"""}

\begin{abstract}
""" + abstract + r"""
\end{abstract}

\begin{document}

\flushbottom
\maketitle
\thispagestyle{empty}

""" + body + r"""

\bibliography{references}

\end{document}
"""
    
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(template)
        
    print(f"[OK] Ported content successfully to {main_path}")

if __name__ == "__main__":
    main()
