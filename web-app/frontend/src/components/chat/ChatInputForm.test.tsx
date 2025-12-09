import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatInputForm from './ChatInputForm';
import { EvaluationConfig } from '../../store/evaluationStore';

describe('ChatInputForm', () => {
  const defaultConfig: EvaluationConfig = {
    judgeModels: ['gpt-4'],
    enableRetrieval: false,
    aggregationStrategy: 'weighted_average',
  };

  const defaultProps = {
    onSubmit: jest.fn(),
    isEvaluating: false,
    config: defaultConfig,
    onConfigChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render form inputs', () => {
    render(<ChatInputForm {...defaultProps} />);
    
    expect(screen.getByLabelText('Source Text')).toBeInTheDocument();
    expect(screen.getByLabelText('Candidate Output')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /evaluate/i })).toBeInTheDocument();
  });

  it('should validate required fields', async () => {
    render(<ChatInputForm {...defaultProps} />);
    
    const submitButton = screen.getByRole('button', { name: /evaluate/i });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText('Source text is required')).toBeInTheDocument();
      expect(screen.getByText('Candidate output is required')).toBeInTheDocument();
    });

    expect(defaultProps.onSubmit).not.toHaveBeenCalled();
  });

  it('should validate minimum length', async () => {
    render(<ChatInputForm {...defaultProps} />);
    
    const sourceInput = screen.getByLabelText('Source Text');
    const candidateInput = screen.getByLabelText('Candidate Output');
    
    await userEvent.type(sourceInput, 'short');
    await userEvent.type(candidateInput, 'short');
    
    const submitButton = screen.getByRole('button', { name: /evaluate/i });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText('Source text must be at least 10 characters')).toBeInTheDocument();
      expect(screen.getByText('Candidate output must be at least 10 characters')).toBeInTheDocument();
    });
  });

  it('should submit valid form', async () => {
    render(<ChatInputForm {...defaultProps} />);
    
    const sourceInput = screen.getByLabelText('Source Text');
    const candidateInput = screen.getByLabelText('Candidate Output');
    
    await userEvent.type(sourceInput, 'This is a valid source text with enough characters');
    await userEvent.type(candidateInput, 'This is a valid candidate output with enough characters');
    
    const submitButton = screen.getByRole('button', { name: /evaluate/i });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(defaultProps.onSubmit).toHaveBeenCalledWith(
        'This is a valid source text with enough characters',
        'This is a valid candidate output with enough characters'
      );
    });
  });

  it('should clear form after submission', async () => {
    render(<ChatInputForm {...defaultProps} />);
    
    const sourceInput = screen.getByLabelText('Source Text') as HTMLTextAreaElement;
    const candidateInput = screen.getByLabelText('Candidate Output') as HTMLTextAreaElement;
    
    await userEvent.type(sourceInput, 'Valid source text here');
    await userEvent.type(candidateInput, 'Valid candidate output here');
    
    const submitButton = screen.getByRole('button', { name: /evaluate/i });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(sourceInput.value).toBe('');
      expect(candidateInput.value).toBe('');
    });
  });

  it('should disable inputs when evaluating', () => {
    render(<ChatInputForm {...defaultProps} isEvaluating={true} />);
    
    const sourceInput = screen.getByLabelText('Source Text');
    const candidateInput = screen.getByLabelText('Candidate Output');
    const submitButton = screen.getByRole('button', { name: /evaluating/i });

    expect(sourceInput).toBeDisabled();
    expect(candidateInput).toBeDisabled();
    expect(submitButton).toBeDisabled();
  });

  it('should toggle configuration panel', async () => {
    render(<ChatInputForm {...defaultProps} />);
    
    const configButton = screen.getByRole('button', { name: /configuration/i });
    
    // Initially hidden
    expect(screen.queryByText('Judge Models (select at least one)')).not.toBeInTheDocument();
    
    // Click to show
    fireEvent.click(configButton);
    await waitFor(() => {
      expect(screen.getByText('Judge Models (select at least one)')).toBeInTheDocument();
    });
    
    // Click to hide
    fireEvent.click(configButton);
    await waitFor(() => {
      expect(screen.queryByText('Judge Models (select at least one)')).not.toBeInTheDocument();
    });
  });

  it('should toggle judge models', async () => {
    render(<ChatInputForm {...defaultProps} />);
    
    // Open config panel
    const configButton = screen.getByRole('button', { name: /configuration/i });
    fireEvent.click(configButton);

    await waitFor(() => {
      const gpt35Button = screen.getByRole('button', { name: 'gpt-3.5-turbo' });
      fireEvent.click(gpt35Button);
    });

    expect(defaultProps.onConfigChange).toHaveBeenCalledWith({
      judgeModels: ['gpt-4', 'gpt-3.5-turbo'],
    });
  });

  it('should not allow deselecting all judge models', async () => {
    render(<ChatInputForm {...defaultProps} />);
    
    // Open config panel
    const configButton = screen.getByRole('button', { name: /configuration/i });
    fireEvent.click(configButton);

    await waitFor(() => {
      const gpt4Button = screen.getByRole('button', { name: 'gpt-4' });
      fireEvent.click(gpt4Button);
    });

    // Should not call onConfigChange with empty array
    expect(defaultProps.onConfigChange).not.toHaveBeenCalled();
  });

  it('should toggle retrieval setting', async () => {
    render(<ChatInputForm {...defaultProps} />);
    
    // Open config panel
    const configButton = screen.getByRole('button', { name: /configuration/i });
    fireEvent.click(configButton);

    await waitFor(() => {
      const retrievalCheckbox = screen.getByLabelText('Enable Retrieval');
      fireEvent.click(retrievalCheckbox);
    });

    expect(defaultProps.onConfigChange).toHaveBeenCalledWith({
      enableRetrieval: true,
    });
  });

  it('should change aggregation strategy', async () => {
    render(<ChatInputForm {...defaultProps} />);
    
    // Open config panel
    const configButton = screen.getByRole('button', { name: /configuration/i });
    fireEvent.click(configButton);

    await waitFor(() => {
      expect(screen.getByText('Judge Models (select at least one)')).toBeInTheDocument();
    });

    const strategySelect = screen.getByDisplayValue('Weighted Average');
    fireEvent.change(strategySelect, { target: { value: 'median' } });

    expect(defaultProps.onConfigChange).toHaveBeenCalledWith({
      aggregationStrategy: 'median',
    });
  });
});
