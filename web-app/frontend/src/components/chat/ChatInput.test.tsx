import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatInput from './ChatInput';

describe('ChatInput', () => {
  const defaultProps = {
    onSubmit: jest.fn(),
    isProcessing: false,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render the input field', () => {
      render(<ChatInput {...defaultProps} />);
      
      expect(screen.getByRole('textbox', { name: /chat input/i })).toBeInTheDocument();
    });

    it('should render the send button', () => {
      render(<ChatInput {...defaultProps} />);
      
      expect(screen.getByRole('button', { name: /send message/i })).toBeInTheDocument();
    });

    it('should render with custom placeholder', () => {
      render(<ChatInput {...defaultProps} placeholder="Type your question here..." />);
      
      expect(screen.getByPlaceholderText('Type your question here...')).toBeInTheDocument();
    });

    it('should render with default placeholder', () => {
      render(<ChatInput {...defaultProps} />);
      
      expect(screen.getByPlaceholderText('Ask a question...')).toBeInTheDocument();
    });
  });

  describe('Send Button State (Requirement 1.4)', () => {
    it('should disable send button when input is empty', () => {
      render(<ChatInput {...defaultProps} />);
      
      const sendButton = screen.getByRole('button', { name: /send message/i });
      expect(sendButton).toBeDisabled();
    });

    it('should disable send button when input contains only whitespace', async () => {
      render(<ChatInput {...defaultProps} />);
      
      const input = screen.getByRole('textbox', { name: /chat input/i });
      await userEvent.type(input, '   ');
      
      const sendButton = screen.getByRole('button', { name: /send message/i });
      expect(sendButton).toBeDisabled();
    });

    it('should enable send button when input has content', async () => {
      render(<ChatInput {...defaultProps} />);
      
      const input = screen.getByRole('textbox', { name: /chat input/i });
      await userEvent.type(input, 'Hello');
      
      const sendButton = screen.getByRole('button', { name: /send message/i });
      expect(sendButton).not.toBeDisabled();
    });
  });

  describe('Processing State', () => {
    it('should disable input when processing', () => {
      render(<ChatInput {...defaultProps} isProcessing={true} />);
      
      const input = screen.getByRole('textbox', { name: /chat input/i });
      expect(input).toBeDisabled();
    });

    it('should disable send button when processing', () => {
      render(<ChatInput {...defaultProps} isProcessing={true} />);
      
      const sendButton = screen.getByRole('button', { name: /send message/i });
      expect(sendButton).toBeDisabled();
    });

    it('should show spinner when processing', () => {
      render(<ChatInput {...defaultProps} isProcessing={true} />);
      
      const sendButton = screen.getByRole('button', { name: /send message/i });
      const spinner = sendButton.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });
  });

  describe('Form Submission (Requirements 1.1, 1.2)', () => {
    it('should call onSubmit with trimmed input when clicking send button', async () => {
      render(<ChatInput {...defaultProps} />);
      
      const input = screen.getByRole('textbox', { name: /chat input/i });
      await userEvent.type(input, '  What is the capital of France?  ');
      
      const sendButton = screen.getByRole('button', { name: /send message/i });
      fireEvent.click(sendButton);
      
      expect(defaultProps.onSubmit).toHaveBeenCalledWith('What is the capital of France?');
    });

    it('should call onSubmit when pressing Enter', async () => {
      render(<ChatInput {...defaultProps} />);
      
      const input = screen.getByRole('textbox', { name: /chat input/i });
      await userEvent.type(input, 'What is AI?');
      await userEvent.keyboard('{Enter}');
      
      expect(defaultProps.onSubmit).toHaveBeenCalledWith('What is AI?');
    });

    it('should NOT call onSubmit when pressing Shift+Enter', async () => {
      render(<ChatInput {...defaultProps} />);
      
      const input = screen.getByRole('textbox', { name: /chat input/i });
      await userEvent.type(input, 'Line 1');
      await userEvent.keyboard('{Shift>}{Enter}{/Shift}');
      
      expect(defaultProps.onSubmit).not.toHaveBeenCalled();
    });

    it('should clear input after submission', async () => {
      render(<ChatInput {...defaultProps} />);
      
      const input = screen.getByRole('textbox', { name: /chat input/i }) as HTMLTextAreaElement;
      await userEvent.type(input, 'Test question');
      
      const sendButton = screen.getByRole('button', { name: /send message/i });
      fireEvent.click(sendButton);
      
      await waitFor(() => {
        expect(input.value).toBe('');
      });
    });

    it('should NOT submit when input is empty', async () => {
      render(<ChatInput {...defaultProps} />);
      
      const sendButton = screen.getByRole('button', { name: /send message/i });
      fireEvent.click(sendButton);
      
      expect(defaultProps.onSubmit).not.toHaveBeenCalled();
    });

    it('should NOT submit when input is only whitespace', async () => {
      render(<ChatInput {...defaultProps} />);
      
      const input = screen.getByRole('textbox', { name: /chat input/i });
      await userEvent.type(input, '   ');
      
      // Try to submit via Enter
      await userEvent.keyboard('{Enter}');
      
      expect(defaultProps.onSubmit).not.toHaveBeenCalled();
    });

    it('should NOT submit when processing', async () => {
      render(<ChatInput {...defaultProps} isProcessing={true} />);
      
      // Input is disabled, but let's verify the button click doesn't work
      const sendButton = screen.getByRole('button', { name: /send message/i });
      fireEvent.click(sendButton);
      
      expect(defaultProps.onSubmit).not.toHaveBeenCalled();
    });
  });

  describe('Accessibility', () => {
    it('should have proper aria-label on input', () => {
      render(<ChatInput {...defaultProps} />);
      
      expect(screen.getByRole('textbox', { name: /chat input/i })).toBeInTheDocument();
    });

    it('should have proper aria-label on send button', () => {
      render(<ChatInput {...defaultProps} />);
      
      expect(screen.getByRole('button', { name: /send message/i })).toBeInTheDocument();
    });
  });
});
