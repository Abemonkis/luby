% File: customjournal.cls
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ijclclp}
\LoadClass[a4paper,10pt]{article}

% Packages
\RequirePackage{amsmath, amssymb, graphicx, fancyhdr, geometry, titlesec, scrextend, xeCJK, natbib, booktabs}

\setCJKmainfont{cwTeXMing}
\setCJKmonofont{cwTeXMing}
\bibliographystyle{apalike}  

% Page Layout
\geometry{
    papersize={190mm,260mm}, top=3.7cm, bottom=2.3cm, left=2.7cm, right=2.6cm}

\deffootnote{3mm}{1.6em}{\thefootnotemark.\enskip}

% Header and Footer
\fancypagestyle{firstpage}{
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{Computational Linguistics and Chinese Language Processing
\\
Vol. xx, No. x, June 2007, pp. xx-xx
\\
\textcopyright \ The Association for Computational Linguistics and Chinese Language Processing}
}
\renewcommand{\headrulewidth}{0pt}
\setlength{\headheight}{36pt}
% Define internal storage for affiliation and email
\newcommand{\AffiliationText}{}
\newcommand{\EmailText}{}

% Define \affiliation and \email commands
\newcommand{\affiliation}[1]{\renewcommand{\AffiliationText}{#1}}
\newcommand{\email}[1]{\renewcommand{\EmailText}{#1}}

% Title Formatting with Footnote for Affiliations and Emails
\renewcommand{\maketitle}{
    \begin{center}
        \vspace*{1cm}
        {\LARGE \bfseries \@title \par}
        \vspace{1em}
        {\large \@author \par}
        \vspace{1em}
        % \@date
    \end{center}
    \ifx\AffiliationText\empty\else
        \footnotetext[1]{\AffiliationText \\  \ifx\EmailText\empty\else
            E-mail: \EmailText 
        \fi}
    \fi

}

% Abstract Formatting
\renewenvironment{abstract}{
    \begin{center}
        \textbf{Abstract}
    \end{center}
    \begin{quote}
}{
    \end{quote}
}

% Section Formatting
\titleformat{\section}
    {\normalfont\Large\bfseries}{\thesection.}{1em}{}

\titleformat{\subsection}
    {\normalfont\large\bfseries}{\thesubsection}{1em}{}

% Keywords Command
\newcommand{\keywords}[1]{%
    \vspace{1em}\noindent\textbf{Keywords:} #1
}

% Begin Document
\endinput
