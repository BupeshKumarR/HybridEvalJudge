import { render, screen } from '@testing-library/react';
import UserMessage from './UserMessage';

describe('UserMessage', () => {
  const defaultProps = {
    sourceText: 'This is the source text',
    candidateOutput: 'This is the candidate output',
    timestamp: new Date('2024-01-01T12:00:00Z'),
  };

  it('should render source text', () => {
    render(<UserMessage {...defaultProps} />);
    
    expect(screen.getByText('Source Text:')).toBeInTheDocument();
    expect(screen.getByText('This is the source text')).toBeInTheDocument();
  });

  it('should render candidate output', () => {
    render(<UserMessage {...defaultProps} />);
    
    expect(screen.getByText('Candidate Output:')).toBeInTheDocument();
    expect(screen.getByText('This is the candidate output')).toBeInTheDocument();
  });

  it('should render timestamp', () => {
    render(<UserMessage {...defaultProps} />);
    
    // The exact text will vary based on current time, so just check it exists
    const timestampElement = screen.getByText(/ago/);
    expect(timestampElement).toBeInTheDocument();
  });

  it('should preserve whitespace in text', () => {
    const propsWithWhitespace = {
      ...defaultProps,
      sourceText: 'Line 1\nLine 2\nLine 3',
    };

    render(<UserMessage {...propsWithWhitespace} />);
    
    const sourceElement = screen.getByText((_content, element) => {
      return element?.textContent === 'Line 1\nLine 2\nLine 3';
    });
    expect(sourceElement).toHaveClass('whitespace-pre-wrap');
  });
});
