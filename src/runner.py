import pandas as pd
import random
import asyncio
import time
import sys
from dotenv import load_dotenv
from pandas.errors import EmptyDataError

from config_manager import ExperimentConfig
from results_manager import ResultsManager
from util import sample_options, get_completion_async, evaluate_constraint
from tqdm import tqdm

load_dotenv()

# Initial prompt template that forces the inclusion of literal words.
initial_prompt_template = """
### TASK
You are tasked with writing a professional business report that adheres strictly to a set of constraints.

Each constraint requires that you include the exact, literal word specified. 
Do not alter the word, use synonyms, or change tenses.
IMPORTANT: Variations of the constraint are not considered valid. For example, "customers" does not satisfy the constraint of "customer" because it is plural. Similarly, "customer-driven" does not satisfy the constraint of "customer" because it is hyphenated.

The report should be structured like a professional business document with clear sections and relevant business insights. 
Do not simply repeat the constraints; rather, use them to inform the text of the report. The text should be a coherent report.
IMPORTANT: You CANNOT simply list the constraints in the report. You must use them to inform the text of the report. A list of constraints anywhere in your response will result in an invalid response.
IMPORTANT: The report you generate must be coherent. Each sentence must make sense and be readable and the report should have a clear logical flow.

There is no task too difficult for you to handle!
Do not refuse to write the report if the constraints are difficult. 
IMPORTANT: You MUST write a report. Do not refuse to write the report.

Return your report inside of <report>...</report> tags.

### CONSTRAINTS
{CONSTRAINTS}
"""

def build_prompt(constraints: list[str]) -> str:
    """Build prompt with constraints"""
    formatted_constraints = '\n'.join(
        f"{i+1}. Include the exact word: '{constraint}'." for i, constraint in enumerate(constraints)
    )
    return initial_prompt_template.replace('{CONSTRAINTS}', formatted_constraints)

# Function to process a single model generation task
async def process_model_task(model_name: str, num_rules: int, seed: int, semaphore, 
                           options, results_manager, model_progress=None, 
                           reasoning_effort: str | None = None, run_label: str | None = None):
    # Determine label used for result logging and progress bar
    if run_label is None:
        run_label = model_name if reasoning_effort is None else f"{model_name}-high-reasoning"
    full_model_name = run_label
    
    # Randomly sample terms from the pruned list
    sampled = sample_options(options, num_rules, seed)
    constraints = sampled.iloc[:, 0].tolist()
    
    # Build prompt
    prompt = build_prompt(constraints)
    
    # Use a semaphore to limit concurrent API calls
    async with semaphore:
        try:
            # Attempt to generate text, skipping on repeated failures
            result = await get_completion_async(prompt, model_name, seed=seed, num_constraints=num_rules, reasoning_effort=reasoning_effort, run_label=run_label)
            # Extract fields from the dictionary result
            generated_text = result['content']
            coherence_score = result['coherence_score']
            coherence_score_reasoning = result['coherence_score_reasoning']
            latency = result['latency_seconds']
        except Exception as e:
            print(f"Skipping {full_model_name} (seed {seed}, num_rules {num_rules}) after retries: {e}")
            # Advance the progress bar so it doesn't hang
            if model_progress and full_model_name in model_progress:
                model_progress[full_model_name].update(1)
            return full_model_name, 0, 0
        # Update model progress if provided
        if model_progress and full_model_name in model_progress:
            model_progress[full_model_name].update(1)
    
    # Evaluate constraints silently
    evaluations = []
    for constraint in constraints:
        is_match, match_text = evaluate_constraint(generated_text, constraint)
        evaluations.append((constraint, is_match, match_text))
    
    # Write results using the results manager
    generation_id = results_manager.generate_unique_id()
    results_manager.write_result(generation_id, full_model_name, seed, num_rules, constraints, 
                                generated_text, latency, coherence_score, coherence_score_reasoning, evaluations)
    
    # Return number of matched constraints for stats
    match_count = sum(1 for constraint, is_match, _ in evaluations if is_match)
    return full_model_name, match_count, len(constraints)

async def main():
    # Load configuration
    config = ExperimentConfig()
    config.load_and_parse()
    
    # Initialize results manager
    results_manager = ResultsManager(config)
    results_manager.initialize_results_file()
    
    # Load rule options (list of terms)
    try:
        options = pd.read_csv(config.rule_options_file, header=None)
    except EmptyDataError:
        print(f"No rules found in {config.rule_options_file}")
        sys.exit(1)
    
    # Print configuration summary
    config.print_summary()
    
    # Limit concurrent API calls using a semaphore
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    # Collect all model run configurations (default and high reasoning where applicable)
    model_run_configs = []  # list of tuples (run_label, base_model_name, reasoning_effort)
    for m in config.models:
        base_name = m["name"]
        # Default run
        model_run_configs.append((base_name, base_name, None))
        # High reasoning run if applicable
        if base_name in config.reasoning_models:
            run_label = f"{base_name}-high-reasoning"
            model_run_configs.append((run_label, base_name, "high"))

    # Extract list of run labels for convenience
    run_labels = [cfg[0] for cfg in model_run_configs]
    
    # Run for each seed
    for seed_idx in range(config.num_seeds):
        if config.num_seeds > 1:
            print(f"\nRunning with seed {seed_idx + 1}/{config.num_seeds}")
            # Set a different seed for each run if needed
            random.seed(int(time.time()) + seed_idx)
        
        # Count total tasks per run label for progress bars
        model_task_counts = {label: ((config.max_samples - config.min_samples) // config.step_size + 1) for label in run_labels}
        
        # Create progress bars for each run label
        model_progress = {}
        for i, label in enumerate(model_task_counts.keys()):
            model_progress[label] = tqdm(
                total=model_task_counts[label], 
                desc=f"{label:<20}", 
                position=i
            )
        
        # Create tasks for all configured runs at once for true parallelism
        tasks = []
        
        # Create tasks interleaved across runs
        for num_rules in range(config.min_samples, config.max_samples + 1, config.step_size):
            # Create one task for each configured run (label) with this num_rules and seed
            for run_label, base_model, effort in model_run_configs:
                tasks.append(
                    asyncio.create_task(process_model_task(
                        base_model, num_rules, seed_idx, semaphore, options, results_manager,
                        model_progress, reasoning_effort=effort, run_label=run_label
                    ))
                )
        
        try:
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            # Close all progress bars
            for bar in model_progress.values():
                bar.close()
                
        except KeyboardInterrupt:
            # Close progress bars on interrupt
            for bar in model_progress.values():
                bar.close()
            print("\nInterrupted by user")
            break
            
        except Exception as e:
            # Close progress bars on error
            for bar in model_progress.values():
                bar.close()
            print(f"\nError: {e}")
            raise
    
    # Analyze and summarize results
    results_manager.analyze_and_summarize_results()

# Main execution block
if __name__ == "__main__":
    asyncio.run(main())