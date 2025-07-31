import pandas as pd
import csv
import os
import uuid
from datetime import datetime
from pandas.errors import EmptyDataError

class ResultsManager:
    def __init__(self, config):
        self.config = config
        self.generations_file = config.generations_file
        self.results_dir = config.results_dir
        self.FIELDNAMES = config.FIELDNAMES
        self.LATENCY_FIELD = config.LATENCY_FIELD
        self.COHERENCE_SCORE_FIELD = config.COHERENCE_SCORE_FIELD
        self.COHERENCE_REASONING_FIELD = config.COHERENCE_REASONING_FIELD
    
    def initialize_results_file(self):
        """Initialize or update the results CSV file"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # If an existing results CSV is provided, drop old runs for these models
        if os.path.exists(self.generations_file):
            print(f"Updating existing results file: {self.generations_file}")
            self._upgrade_generations_csv_header(self.generations_file)
            try:
                existing = pd.read_csv(self.generations_file)
            except EmptyDataError:
                existing = pd.DataFrame(columns=self.FIELDNAMES)
            # Build list of model names to remove (including reasoning efforts)
            drop_models = []
            for model_info in self.config.models:
                base = model_info['name']
                drop_models.append(base)
            # Filter out rows for models being re-run
            existing = existing[~existing['model'].isin(drop_models)]
            existing.to_csv(self.generations_file, index=False)
        else:
            print(f"Creating new results file: {self.generations_file}")
            with open(self.generations_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
    
    def _upgrade_generations_csv_header(self, file_path: str):
        """Helper to ensure an existing generations.csv contains required columns"""
        if not os.path.exists(file_path):
            return

        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return

        header = rows[0]
        
        # Check which fields are missing
        missing_fields = []
        for field in [self.LATENCY_FIELD, self.COHERENCE_SCORE_FIELD, self.COHERENCE_REASONING_FIELD]:
            if field not in header:
                missing_fields.append(field)
        
        if not missing_fields:
            return  # Already up-to-date

        new_header = header + missing_fields
        # Append empty values for missing fields to existing rows
        updated_rows = [new_header]
        for row in rows[1:]:
            updated_rows.append(row + [""] * len(missing_fields))

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(updated_rows)
    
    def write_result(self, generation_id: str, model: str, seed: int, num_rules: int,
                     constraints: list[str], generated_text: str,
                     latency_seconds: float,
                     coherence_score: int,
                     coherence_score_reasoning: str,
                     evaluations: list[tuple[str, bool, str]]):
        """
        Write results to generations.csv with full path.
        """
        with open(self.generations_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow({
                'id': generation_id,
                'model': model,
                'seed': seed,
                'num_rules': num_rules,
                'constraints': ','.join(constraints),
                'generated_text': generated_text,
                'latency_seconds': latency_seconds,
                'coherence_score': coherence_score,
                'coherence_score_reasoning': coherence_score_reasoning,
                'timestamp': datetime.now().isoformat()
            })
    
    def generate_unique_id(self):
        """Generate a unique ID for a result entry"""
        return str(uuid.uuid4())
    
    def analyze_and_summarize_results(self):
        """Analyze results and print summary statistics"""
        print("\nResults Summary:")
        # Read the generations file to get all results
        try:
            df = pd.read_csv(self.generations_file)
        except EmptyDataError:
            print("No results to summarize.")
            return
        
        # Aggregate results by model
        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            # Calculate constraint match counts
            match_data = []
            for _, row in model_df.iterrows():
                constraints = row['constraints'].split(',')
                text = row['generated_text']
                matches = sum(1 for c in constraints if c.lower() in text.lower())
                match_data.append({'matches': matches, 'total': len(constraints)})
            
            total_matches = sum(item['matches'] for item in match_data)
            total_constraints = sum(item['total'] for item in match_data)
            match_rate = total_matches / total_constraints * 100 if total_constraints > 0 else 0
            
            # Calculate coherence statistics
            coherence_scores = model_df[self.COHERENCE_SCORE_FIELD].dropna()
            if len(coherence_scores) > 0:
                avg_coherence = coherence_scores.mean()
                min_coherence = coherence_scores.min()
                max_coherence = coherence_scores.max()
                coherence_info = f", Coherence: {avg_coherence:.1f} (min: {min_coherence}, max: {max_coherence})"
            else:
                coherence_info = ""
                
            print(f"{model_name}: {match_rate:.2f}% constraints matched ({total_matches}/{total_constraints}){coherence_info}")

        # Write per-seed CSV files
        self._write_per_seed_files(df)
    
    def _write_per_seed_files(self, df):
        """Write separate CSV files for each seed"""
        print("\nWriting per-seed result files:")
        result_dir = os.path.dirname(self.generations_file)
        base_name = os.path.splitext(os.path.basename(self.generations_file))[0]
        for seed in df['seed'].unique():
            seed_file = os.path.join(result_dir, f"{base_name}_seed_{seed}.csv")
            df[df['seed'] == seed].to_csv(seed_file, index=False)
            print(f"  Seed {seed}: {seed_file}") 