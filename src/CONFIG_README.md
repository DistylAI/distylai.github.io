# Configuration Guide

The if-scale benchmark uses a configuration file system that makes it easy to customize experiments without modifying code.

## Configuration Files

### Main Configuration (`config.json`)
The main configuration file contains all experiment parameters, model lists, and settings:

```json
{
  "experiment_params": {
    "min_samples": 10,        // Minimum number of constraints
    "max_samples": 500,       // Maximum number of constraints
    "num_seeds": 5,           // Number of different seeds to test
    "step_size": 10,          // Step size between constraint counts
    "max_concurrent_requests": 8  // Max parallel API calls
  },
  "models": [
    {"name": "openai/gpt-4o-mini"},
    {"name": "anthropic/claude-3.5-haiku"}
  ],
  "reasoning_models": [],     // Models to also run with high reasoning effort
  "paths": {
    "results_dir": "outputs/results",
    "default_rules_csv": "inputs/pruned_rules.csv"
  },
  "api": {
    "max_retries": 10,        // Max retries per API call
    "retry_delay_seconds": 3, // Delay between retries
    "min_word_count": 20,     // Minimum response length
    "min_coherence_score": 6  // Minimum coherence score to accept
  }
}
```

## Usage Examples

### 1. Run with Default Configuration
```bash
python runner.py
```

### 2. Run a Single Model
```bash
python runner.py --models "openai/gpt-4o-mini"
```

### 3. Run Multiple Specific Models
```bash
python runner.py --models "openai/gpt-4o-mini" "anthropic/claude-3.5-haiku"
```

### 4. Override Parameters via CLI
```bash
python runner.py --min_samples 20 --max_samples 200 --num_seeds 3 --step_size 20
```

### 5. Use Custom Configuration File
```bash
python runner.py --config config_single_model_example.json
```

### 6. Combine Custom Config with CLI Overrides
```bash
python runner.py --config config_single_model_example.json --models "openai/gpt-4o"
```

## Creating Custom Configurations

1. Copy `config.json` to a new file (e.g., `my_config.json`)
2. Modify the parameters as needed
3. Run with `--config my_config.json`

### Example: Quick Test Configuration
See `config_single_model_example.json` for a configuration that:
- Tests only one model (`openai/gpt-4o-mini`)
- Uses smaller ranges (10-100 constraints)
- Uses fewer seeds (3 instead of 5)
- Reduces concurrent requests for stability

## Configuration Parameters

### Experiment Parameters
- `min_samples`: Starting number of constraints
- `max_samples`: Maximum number of constraints  
- `num_seeds`: Number of random seeds to test
- `step_size`: Increment between constraint counts
- `max_concurrent_requests`: Number of parallel API calls

### Models
- `models`: List of models to test (format: `{"name": "provider/model-name"}`)
- `reasoning_models`: Models that should also run with high reasoning effort

### API Settings
- `max_retries`: Maximum retry attempts for failed API calls
- `retry_delay_seconds`: Seconds to wait between retries
- `min_word_count`: Minimum word count for valid responses
- `min_coherence_score`: Minimum coherence score (1-10) to accept

### Paths
- `results_dir`: Directory for output files
- `default_rules_csv`: Default path to rules/constraints file

## CLI Arguments

- `--config`: Path to custom config file
- `--models`: Override models list
- `--min_samples`, `--max_samples`, `--num_seeds`, `--step_size`: Override experiment parameters
- `--results_csv`: Custom output file path
- `--rules_csv`: Custom rules/constraints file path

CLI arguments always take precedence over config file settings. 