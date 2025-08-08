import argparse
import json
import os
from datetime import datetime
from util import load_config

class ExperimentConfig:
    def __init__(self):
        self.config = None
        self.args = None
        self.models = None
        self.reasoning_models = None
        self.generations_file = None
        self.rule_options_file = None
        self.results_dir = None
        
        # Experiment parameters
        self.min_samples = None
        self.max_samples = None
        self.num_seeds = None
        self.step_size = None
        self.max_concurrent_requests = None
        
        # Field names
        self.LATENCY_FIELD = 'latency_seconds'
        self.COHERENCE_SCORE_FIELD = 'coherence_score'
        self.COHERENCE_REASONING_FIELD = 'coherence_score_reasoning'
        self.FIELDNAMES = None
    
    def load_and_parse(self):
        """Load configuration and parse CLI arguments"""
        # Load base configuration
        self.config = load_config()
        
        # Parse CLI arguments
        self._parse_cli_args()
        
        # Load custom config if specified
        if self.args.config:
            with open(self.args.config, 'r') as f:
                self.config = json.load(f)
        
        # Set up parameters with CLI overrides
        self._setup_parameters()
        
        # Set up paths
        self._setup_paths()
        
        # Determine models to run
        self._setup_models()
        
        # Set up field names
        self.FIELDNAMES = self.config['logging']['generation_fieldnames']
    
    def _parse_cli_args(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description="Run generation experiments")
        parser.add_argument("--results_csv", type=str, default=None,
                            help="Path to results CSV file (appends or overwrites model runs)")
        parser.add_argument("--rules_csv", type=str, default=None,
                            help="Path to pruned rules CSV")
        parser.add_argument("--models", type=str, nargs='+', default=None,
                            help="Specific model names to run (overrides config)")
        parser.add_argument("--min_samples", type=int, default=None,
                            help="Override minimum number of samples")
        parser.add_argument("--max_samples", type=int, default=None,
                            help="Override maximum number of samples")
        parser.add_argument("--num_seeds", type=int, default=None,
                            help="Override number of seeds")
        parser.add_argument("--step_size", type=int, default=None,
                            help="Override step size")
        parser.add_argument("--config", type=str, default=None,
                            help="Path to custom config file")
        self.args = parser.parse_args()
    
    def _setup_parameters(self):
        """Set up experiment parameters with CLI overrides"""
        # Get base parameters from config
        self.min_samples = self.config['experiment_params']['min_samples']
        self.max_samples = self.config['experiment_params']['max_samples']
        self.num_seeds = self.config['experiment_params']['num_seeds']
        self.step_size = self.config['experiment_params']['step_size']
        self.max_concurrent_requests = self.config['experiment_params']['max_concurrent_requests']
        
        # Override with CLI arguments if provided
        if self.args.min_samples is not None:
            self.min_samples = self.args.min_samples
        if self.args.max_samples is not None:
            self.max_samples = self.args.max_samples
        if self.args.num_seeds is not None:
            self.num_seeds = self.args.num_seeds
        if self.args.step_size is not None:
            self.step_size = self.args.step_size
    
    def _setup_paths(self):
        """Set up file paths"""
        base_dir = os.path.abspath(os.path.dirname(__file__))
        self.results_dir = os.path.abspath(os.path.join(base_dir, self.config['paths']['results_dir']))
        
        # Determine results file
        if self.args.results_csv:
            self.generations_file = os.path.abspath(self.args.results_csv)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.generations_file = os.path.abspath(os.path.join(self.results_dir, f'generations_{timestamp}.csv'))
        
        # Determine rules CSV
        if self.args.rules_csv:
            self.rule_options_file = os.path.abspath(self.args.rules_csv)
        else:
            self.rule_options_file = os.path.abspath(os.path.join(base_dir, self.config['paths']['default_rules_csv']))
    
    def _setup_models(self):
        """Determine which models to run"""
        if self.args.models:
            # Use specified models from CLI
            self.models = [{"name": model} for model in self.args.models]
            self.reasoning_models = []  # Don't run reasoning models unless specifically in config
        else:
            # Use models from config
            self.models = self.config['models']
            self.reasoning_models = self.config['reasoning_models']
    
    def print_summary(self):
        """Print configuration summary"""
        print(f"Testing {len(self.models)} models with rules from {self.min_samples} to {self.max_samples} (step size {self.step_size})")
        print(f"Using {self.num_seeds} seeds")
        print(f"Results will be written to: {self.generations_file}")
        print(f"Using rules from: {self.rule_options_file}")
        
        if self.args.models:
            print(f"Running specific models: {self.args.models}") 