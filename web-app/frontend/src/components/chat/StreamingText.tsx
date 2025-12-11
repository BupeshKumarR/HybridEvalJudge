import React, { useEffect, useState, useRef, useMemo } from 'react';

interface StreamingTextProps {
  /** The text content to display, updated as tokens arrive */
  text: string;
  /** Whether streaming is currently in progress */
  isStreaming: boolean;
  /** Optional CSS class for the text container */
  className?: string;
}

/**
 * Render LaTeX math expression to HTML using basic substitutions
 * This provides readable math without requiring external libraries
 */
const renderLatexToHtml = (latex: string): string => {
  let html = latex.trim();
  
  // Handle common LaTeX commands with HTML/Unicode equivalents
  const replacements: [RegExp, string][] = [
    // Greek letters
    [/\\alpha/g, 'α'], [/\\beta/g, 'β'], [/\\gamma/g, 'γ'], [/\\delta/g, 'δ'],
    [/\\epsilon/g, 'ε'], [/\\zeta/g, 'ζ'], [/\\eta/g, 'η'], [/\\theta/g, 'θ'],
    [/\\iota/g, 'ι'], [/\\kappa/g, 'κ'], [/\\lambda/g, 'λ'], [/\\mu/g, 'μ'],
    [/\\nu/g, 'ν'], [/\\xi/g, 'ξ'], [/\\pi/g, 'π'], [/\\rho/g, 'ρ'],
    [/\\sigma/g, 'σ'], [/\\tau/g, 'τ'], [/\\upsilon/g, 'υ'], [/\\phi/g, 'φ'],
    [/\\chi/g, 'χ'], [/\\psi/g, 'ψ'], [/\\omega/g, 'ω'],
    [/\\Gamma/g, 'Γ'], [/\\Delta/g, 'Δ'], [/\\Theta/g, 'Θ'], [/\\Lambda/g, 'Λ'],
    [/\\Xi/g, 'Ξ'], [/\\Pi/g, 'Π'], [/\\Sigma/g, 'Σ'], [/\\Phi/g, 'Φ'],
    [/\\Psi/g, 'Ψ'], [/\\Omega/g, 'Ω'],
    
    // Math operators and symbols
    [/\\infty/g, '∞'], [/\\partial/g, '∂'], [/\\nabla/g, '∇'],
    [/\\pm/g, '±'], [/\\mp/g, '∓'], [/\\times/g, '×'], [/\\div/g, '÷'],
    [/\\cdot/g, '·'], [/\\ast/g, '∗'], [/\\star/g, '⋆'],
    [/\\leq/g, '≤'], [/\\geq/g, '≥'], [/\\neq/g, '≠'], [/\\approx/g, '≈'],
    [/\\equiv/g, '≡'], [/\\sim/g, '∼'], [/\\propto/g, '∝'],
    [/\\subset/g, '⊂'], [/\\supset/g, '⊃'], [/\\subseteq/g, '⊆'], [/\\supseteq/g, '⊇'],
    [/\\in/g, '∈'], [/\\notin/g, '∉'], [/\\ni/g, '∋'],
    [/\\cup/g, '∪'], [/\\cap/g, '∩'], [/\\emptyset/g, '∅'],
    [/\\forall/g, '∀'], [/\\exists/g, '∃'], [/\\neg/g, '¬'],
    [/\\wedge/g, '∧'], [/\\vee/g, '∨'], [/\\oplus/g, '⊕'], [/\\otimes/g, '⊗'],
    [/\\rightarrow/g, '→'], [/\\leftarrow/g, '←'], [/\\Rightarrow/g, '⇒'],
    [/\\Leftarrow/g, '⇐'], [/\\leftrightarrow/g, '↔'], [/\\Leftrightarrow/g, '⇔'],
    [/\\to/g, '→'], [/\\gets/g, '←'], [/\\mapsto/g, '↦'],
    [/\\sum/g, '∑'], [/\\prod/g, '∏'], [/\\int/g, '∫'],
    [/\\sqrt/g, '√'], [/\\ldots/g, '…'], [/\\cdots/g, '⋯'],
    
    // Spacing and formatting
    [/\\quad/g, '  '], [/\\qquad/g, '    '], [/\\,/g, ' '], [/\\;/g, ' '],
    [/\\!/g, ''], [/\\ /g, ' '],
    [/\\left\s*\(/g, '('], [/\\right\s*\)/g, ')'],
    [/\\left\s*\[/g, '['], [/\\right\s*\]/g, ']'],
    [/\\left\s*\{/g, '{'], [/\\right\s*\}/g, '}'],
    [/\\left\s*\|/g, '|'], [/\\right\s*\|/g, '|'],
    [/\\{/g, '{'], [/\\}/g, '}'],
    [/\\langle/g, '⟨'], [/\\rangle/g, '⟩'],
  ];
  
  for (const [pattern, replacement] of replacements) {
    html = html.replace(pattern, replacement);
  }
  
  // Handle \mathcal{X} -> 𝒳 (script letters)
  html = html.replace(/\\mathcal\{([A-Z])\}/g, (_, letter) => {
    const scriptMap: Record<string, string> = {
      'A': '𝒜', 'B': 'ℬ', 'C': '𝒞', 'D': '𝒟', 'E': 'ℰ', 'F': 'ℱ', 'G': '𝒢',
      'H': 'ℋ', 'I': 'ℐ', 'J': '𝒥', 'K': '𝒦', 'L': 'ℒ', 'M': 'ℳ', 'N': '𝒩',
      'O': '𝒪', 'P': '𝒫', 'Q': '𝒬', 'R': 'ℛ', 'S': '𝒮', 'T': '𝒯', 'U': '𝒰',
      'V': '𝒱', 'W': '𝒲', 'X': '𝒳', 'Y': '𝒴', 'Z': '𝒵'
    };
    return scriptMap[letter] || letter;
  });
  
  // Handle \mathbb{X} -> 𝕏 (blackboard bold)
  html = html.replace(/\\mathbb\{([A-Z])\}/g, (_, letter) => {
    const bbMap: Record<string, string> = {
      'A': '𝔸', 'B': '𝔹', 'C': 'ℂ', 'D': '𝔻', 'E': '𝔼', 'F': '𝔽', 'G': '𝔾',
      'H': 'ℍ', 'I': '𝕀', 'J': '𝕁', 'K': '𝕂', 'L': '𝕃', 'M': '𝕄', 'N': 'ℕ',
      'O': '𝕆', 'P': 'ℙ', 'Q': 'ℚ', 'R': 'ℝ', 'S': '𝕊', 'T': '𝕋', 'U': '𝕌',
      'V': '𝕍', 'W': '𝕎', 'X': '𝕏', 'Y': '𝕐', 'Z': 'ℤ'
    };
    return bbMap[letter] || letter;
  });
  
  // Handle fractions: \frac{a}{b} -> a/b or (a)/(b)
  html = html.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, '($1)/($2)');
  
  // Handle subscripts: _{x} -> ₓ or _x -> ₓ
  html = html.replace(/_\{([^{}]+)\}/g, '<sub>$1</sub>');
  html = html.replace(/_([a-zA-Z0-9])/g, '<sub>$1</sub>');
  
  // Handle superscripts: ^{x} -> ˣ or ^x -> ˣ  
  html = html.replace(/\^\{([^{}]+)\}/g, '<sup>$1</sup>');
  html = html.replace(/\^([a-zA-Z0-9])/g, '<sup>$1</sup>');
  
  // Handle \log_x -> log with subscript
  html = html.replace(/\\log_([a-zA-Z0-9])/g, 'log<sub>$1</sub>');
  html = html.replace(/\\log_\{([^{}]+)\}/g, 'log<sub>$1</sub>');
  html = html.replace(/\\log/g, 'log');
  html = html.replace(/\\ln/g, 'ln');
  html = html.replace(/\\exp/g, 'exp');
  html = html.replace(/\\sin/g, 'sin');
  html = html.replace(/\\cos/g, 'cos');
  html = html.replace(/\\tan/g, 'tan');
  
  // Handle matrices: \begin{bmatrix}...\end{bmatrix}
  html = html.replace(/\\begin\{bmatrix\}([\s\S]*?)\\end\{bmatrix\}/g, (_, content) => {
    const rows = content.split('\\\\').map((row: string) => 
      row.split('&').map((cell: string) => cell.trim()).join(' | ')
    ).join(' ⟩ ⟨ ');
    return `[ ${rows} ]`;
  });
  
  html = html.replace(/\\begin\{pmatrix\}([\s\S]*?)\\end\{pmatrix\}/g, (_, content) => {
    const rows = content.split('\\\\').map((row: string) => 
      row.split('&').map((cell: string) => cell.trim()).join(' | ')
    ).join(' ⟩ ⟨ ');
    return `( ${rows} )`;
  });
  
  // Clean up remaining backslashes from unknown commands
  html = html.replace(/\\([a-zA-Z]+)/g, '$1');
  
  return html;
};

/**
 * Simple markdown renderer for basic formatting
 * Handles: **bold**, *italic*, `code`, ```code blocks```, LaTeX math, and line breaks
 */
const renderMarkdown = (text: string): React.ReactNode[] => {
  if (!text) return [];

  const elements: React.ReactNode[] = [];
  let key = 0;

  // Split by code blocks first
  const codeBlockRegex = /```(\w*)\n?([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  const processInlineMarkdown = (str: string): React.ReactNode[] => {
    const inlineElements: React.ReactNode[] = [];
    let inlineKey = 0;

    // Process LaTeX math ($...$ and $$...$$), bold, italic, and inline code
    // Also catch raw LaTeX commands like \mathcal{T}_x
    const inlineRegex = /(\$\$([^$]+)\$\$|\$([^$\n]+)\$|(\\[a-zA-Z]+(?:\{[^}]*\})*(?:\s*[_^]\s*(?:\{[^}]*\}|[a-zA-Z0-9]))*)|\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`)/g;
    let inlineLastIndex = 0;
    let inlineMatch;

    while ((inlineMatch = inlineRegex.exec(str)) !== null) {
      // Add text before the match
      if (inlineMatch.index > inlineLastIndex) {
        inlineElements.push(str.slice(inlineLastIndex, inlineMatch.index));
      }

      if (inlineMatch[2]) {
        // Display math ($$...$$)
        const renderedMath = renderLatexToHtml(inlineMatch[2]);
        inlineElements.push(
          <div
            key={`display-math-${inlineKey++}`}
            className="my-3 py-3 px-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg overflow-x-auto text-center"
          >
            <span 
              className="text-lg font-serif text-blue-900"
              dangerouslySetInnerHTML={{ __html: renderedMath }}
            />
          </div>
        );
      } else if (inlineMatch[3]) {
        // Inline math ($...$)
        const renderedMath = renderLatexToHtml(inlineMatch[3]);
        inlineElements.push(
          <span
            key={`inline-math-${inlineKey++}`}
            className="bg-blue-50 text-blue-900 px-1.5 py-0.5 rounded border border-blue-200 font-serif mx-0.5"
            dangerouslySetInnerHTML={{ __html: renderedMath }}
          />
        );
      } else if (inlineMatch[4]) {
        // Raw LaTeX command (e.g., \mathcal{T}_x)
        const renderedMath = renderLatexToHtml(inlineMatch[4]);
        inlineElements.push(
          <span
            key={`raw-latex-${inlineKey++}`}
            className="bg-blue-50 text-blue-900 px-1.5 py-0.5 rounded border border-blue-200 font-serif mx-0.5"
            dangerouslySetInnerHTML={{ __html: renderedMath }}
          />
        );
      } else if (inlineMatch[5]) {
        // Bold
        inlineElements.push(
          <strong key={`bold-${inlineKey++}`} className="font-semibold">
            {inlineMatch[5]}
          </strong>
        );
      } else if (inlineMatch[6]) {
        // Italic
        inlineElements.push(
          <em key={`italic-${inlineKey++}`} className="italic">
            {inlineMatch[6]}
          </em>
        );
      } else if (inlineMatch[7]) {
        // Inline code
        inlineElements.push(
          <code
            key={`code-${inlineKey++}`}
            className="bg-gray-200 text-gray-800 px-1 py-0.5 rounded text-sm font-mono"
          >
            {inlineMatch[7]}
          </code>
        );
      }

      inlineLastIndex = inlineMatch.index + inlineMatch[0].length;
    }

    // Add remaining text
    if (inlineLastIndex < str.length) {
      inlineElements.push(str.slice(inlineLastIndex));
    }

    return inlineElements.length > 0 ? inlineElements : [str];
  };

  while ((match = codeBlockRegex.exec(text)) !== null) {
    // Add text before the code block
    if (match.index > lastIndex) {
      const beforeText = text.slice(lastIndex, match.index);
      const lines = beforeText.split('\n');
      lines.forEach((line, i) => {
        elements.push(...processInlineMarkdown(line));
        if (i < lines.length - 1) {
          elements.push(<br key={`br-${key++}`} />);
        }
      });
    }

    // Add the code block
    const language = match[1] || '';
    const code = match[2].trim();
    elements.push(
      <pre
        key={`codeblock-${key++}`}
        className="bg-gray-800 text-gray-100 p-3 rounded-lg my-2 overflow-x-auto text-sm font-mono"
      >
        {language && (
          <div className="text-xs text-gray-400 mb-2">{language}</div>
        )}
        <code>{code}</code>
      </pre>
    );

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text after last code block
  if (lastIndex < text.length) {
    const remainingText = text.slice(lastIndex);
    const lines = remainingText.split('\n');
    lines.forEach((line, i) => {
      elements.push(...processInlineMarkdown(line));
      if (i < lines.length - 1) {
        elements.push(<br key={`br-${key++}`} />);
      }
    });
  }

  return elements;
};

/**
 * StreamingText - Displays text tokens as they arrive with a typing indicator
 *
 * Features:
 * - Display tokens as they arrive
 * - Show typing indicator during streaming
 * - Smooth text accumulation
 * - Basic markdown rendering (bold, italic, code)
 * - LaTeX math rendering with Unicode symbols
 *
 * Requirements: 2.2, 2.5
 */
const StreamingText: React.FC<StreamingTextProps> = ({
  text,
  isStreaming,
  className = '',
}) => {
  const [displayedText, setDisplayedText] = useState('');
  const textRef = useRef<HTMLDivElement>(null);
  const previousTextRef = useRef('');

  useEffect(() => {
    if (text.length > previousTextRef.current.length) {
      setDisplayedText(text);
    } else if (text !== previousTextRef.current) {
      setDisplayedText(text);
    }
    previousTextRef.current = text;
  }, [text]);

  useEffect(() => {
    if (isStreaming && textRef.current) {
      textRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [displayedText, isStreaming]);

  const renderedContent = useMemo(() => {
    if (isStreaming) {
      return (
        <span className="whitespace-pre-wrap break-words">{displayedText}</span>
      );
    }
    return <div className="break-words">{renderMarkdown(displayedText)}</div>;
  }, [displayedText, isStreaming]);

  return (
    <div ref={textRef} className={`relative ${className}`}>
      {renderedContent}
      {isStreaming && (
        <span
          className="inline-block ml-0.5 animate-pulse"
          aria-label="Typing indicator"
        >
          <span className="inline-block w-2 h-4 bg-current opacity-70 rounded-sm" />
        </span>
      )}
    </div>
  );
};

export default StreamingText;
