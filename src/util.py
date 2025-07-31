import pandas as pd
import random
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import re
import asyncio
import json
import uuid
import csv
from datetime import datetime
import aiohttp
import time

load_dotenv()

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

config = load_config()

# Available models (loaded from config)
models = config['models']

# List of models that will also be run with high reasoning effort
reasoning_models = config['reasoning_models']

def sample_options(options: pd.DataFrame, num_rules: int, seed: int) -> pd.DataFrame:
    """
    Randomly sample a specified number of rules from a DataFrame of options.
    
    :param options: DataFrame containing rule options.
    :param num_rules: Number of rules to sample.
    :param seed: Seed for randomization.
    :return: DataFrame with sampled rules.
    """
    if seed is not None:
        random.seed(seed)
    return options.sample(n=num_rules, random_state=seed)

def evaluate_constraint(text: str, constraint: str) -> tuple[bool, str]:
    """
    Deterministically evaluate if a text contains an exact word match.
    Supports '*' wildcards in the constraint.
    
    :param text: The text to check.
    :param constraint: The word pattern to find, can contain asterisks as wildcards.
    :return: Tuple of (success, match_text).
    """
    # Remove Markdown emphasis markup:  *word*, **word**, _word_, __word__
    text = re.sub(r'(\*{1,2}|_{1,2})(.*?)\1', r'\2', text)

    # Split on whitespace only; hyphens and plurals are not modified
    words = text.lower().split()
    constraint = constraint.lower()
    
    # Build regex pattern for exact match with wildcard support
    regex_pattern = '^' + ''.join([
        c if c != '*' else '.*' for c in re.escape(constraint).replace('\\*', '*')
    ]) + '$'
    try:
        pattern = re.compile(regex_pattern)
    except re.error:
        return False, ''
    
    for i, token in enumerate(words):
        clean_token = token.strip('.,!?()[]{}":;*_')
        if pattern.match(clean_token):
            start = max(0, i - 5)
            end = min(len(words), i + 6)
            context = ' '.join(words[start:end])
            return True, context
    return False, ''

# Add unified OpenRouter client and call functions
openrouter_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Add path constants and helper to initialize attempts_log
base_dir = os.path.abspath(os.path.dirname(__file__))
results_dir = os.path.abspath(os.path.join(base_dir, config['paths']['results_dir']))
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

ATTEMPT_LOG = os.path.join(results_dir, 'attempts_log.csv')
ATTEMPT_FIELDNAMES = config['logging']['attempt_fieldnames']

# Ensure the attempts log has a header exactly once
if not os.path.exists(ATTEMPT_LOG):
    with open(ATTEMPT_LOG, 'w', newline='', encoding='utf-8') as _f:
        writer = csv.DictWriter(_f, fieldnames=ATTEMPT_FIELDNAMES)
        writer.writeheader()

def _log_attempt(model: str, seed: int, num_rules: int, attempt_idx: int, status: str, reason: str, coherence_score: int | None = None):
    """Write a single row capturing an attempt outcome (success, rejected, or error)."""
    with open(ATTEMPT_LOG, 'a', newline='', encoding='utf-8') as _f:
        writer = csv.DictWriter(_f, fieldnames=ATTEMPT_FIELDNAMES)
        writer.writerow({
            'id': str(uuid.uuid4()),
            'model': model,
            'seed': seed,
            'num_rules': num_rules,
            'attempt': attempt_idx,
            'status': status,  # success | rejected | error
            'reason': reason,
            'coherence_score': coherence_score if coherence_score is not None else '',
            'timestamp': datetime.now().isoformat()
        })

async def call_model(prompt: str, model_name: str, seed: int = None, num_constraints: int = None, reasoning_effort: str | None = None, run_label: str | None = None) -> dict:
    """Call the OpenRouter client with retries on failure or short responses.
    
    Returns:
        dict: Contains 'content', 'coherence_score', 'coherence_score_reasoning', and 'latency_seconds'
    """
    messages = [{"role": "user", "content": prompt}]
    max_retries = config['api']['max_retries']
    retry_delay = config['api']['retry_delay_seconds']
    min_word_count = config['api']['min_word_count']
    min_coherence_score = config['api']['min_coherence_score']
    
    for attempt in range(1, max_retries + 1):
        try:
            # Determine label used for logging/results (differentiate reasoning runs)
            model_label = run_label if run_label else model_name

            # Time only the successful API call
            api_start_time = time.time()
            
            # Branch on whether we should apply high reasoning effort
            if reasoning_effort is not None:
                # Use raw HTTP call with aiohttp to pass reasoning parameter
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "reasoning": {
                        "effort": reasoning_effort
                    }
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"API request failed with status {response.status}: {error_text}")
                        resp_data = await response.json()
                        content = resp_data['choices'][0]['message']['content']
            else:
                # Use OpenAI SDK for non-reasoning calls
                resp = await openrouter_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
                content = resp.choices[0].message.content
            
            # Record latency for this successful API call
            api_latency = time.time() - api_start_time

            # Detect cheating list: 10+ single words separated by commas
            parts = [p.strip() for p in content.split(',')]
            if len(parts) >= 10 and all(re.fullmatch(r"[A-Za-z]+", p) for p in parts):
                print(f"Cheating list detected from {model_name} (num_constraints={num_constraints}, seed={seed}, attempt {attempt}/{max_retries}), retrying...")
                await asyncio.sleep(retry_delay)
                _log_attempt(model_label, seed, num_constraints, attempt, 'rejected', 'cheating_list')
                continue
            # Check word count
            word_count = len(content.split())
            if word_count < min_word_count:
                print(f"Short response ({word_count} words) from {model_name} (num_constraints={num_constraints}, seed={seed}, attempt {attempt}/{max_retries}), retrying...")
                await asyncio.sleep(retry_delay)
                _log_attempt(model_label, seed, num_constraints, attempt, 'rejected', 'short_response')
                continue
            # Coherence validation via o4-mini
            try:
                validation_prompt = """You are evaluating whether a given professional business report is coherent. Use the following rubric in order to evaluate coherence.
### Coherence Rubric

Use the descriptors below to judge how coherent a business report is **purely on writing quality and logic**, not on whether its facts are backed by evidence.  

| Score | One-line label           | Sentence-level clarity                                                | Logical / causal flow                                            | Domain consistency                                                             | Typical red-flags you'd still see                                              |
|-------|--------------------------|-----------------------------------------------------------------------|------------------------------------------------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| **10**| *Pristine*               | Every sentence is plain-English clear; jargon is either absent or explicitly defined. | Arguments unfold step-by-step; no gaps.                           | Stays in one domain (or clearly signals and justifies any shift).              | Minor copy-editing glitches only.                                              |
| **9** | *Fully coherent*         | 95 %+ of sentences are clear; any buzzwords are easy to decode.       | Tight narrative with just an occasional weak connective phrase.  | Domain focus maintained; at most one brief tangent.                            | Isolated over-statements (e.g., "blockchain eliminates delays").               |
| **8** | *Very strong*            | Sentences are readable but some rely on industry shorthand.           | Flow solid, though 1–2 transitions feel rushed or implied.        | Mostly single-domain; brief forays elsewhere are labelled.                     | A few mild cause-effect leaps.                                                 |
| **7** | *Good with blemishes*    | Majority of sentences clear, but a noticeable handful need re-reading.| Structure makes sense, yet several paragraphs feel loosely stitched.| 1–2 domain jumps without warning.                                              | Buzzword stuffing or vague "synergy" statements appear occasionally.           |
| **6** | *Borderline solid*       | Clarity and vagueness are roughly 60 / 40.                            | Core argument present but dotted with missing steps or circular logic. | Drifts across domains enough to cause momentary confusion.                     | Repeated filler phrases ("next-gen", "holistic transformation").               |
| **5** | *Patchy / mixed*         | Clear and muddled sentences are roughly equal.                        | Reader must infer several causal links; outline feels choppy.     | Multiple domain shifts, sometimes within a single paragraph.                   | Undefined jargon shows up often; occasional contradictions.                    |
| **4** | *Weak*                   | < 50 % of sentences are easily intelligible.                          | Sections read like bullet-lists stapled together; flow is erratic.| Domains regularly collide (finance + biotech + HR) with no bridge.             | Heavy "consultant-speak" and random buzzword chains.                           |
| **3** | *Disjointed*             | Most sentences syntactically valid but stuffed with unrelated clauses.| Logical through-line hard to locate; paragraphs feel random.      | Constant, unexplained domain hopping.                                          | Whole sentences read as word salad; writer seems unaware of terms used.        |
| **2** | *Barely business-like*   | Syntax intact, but meaning opaque; jargon dominates.                  | Almost no causal linkage; ordering seems arbitrary.              | Topic drifts wildly; sections don't build on each other.                       | Many clauses are outright non-sequitur.                                        |
| **1** | *Total gibberish*        | Grammar frequently broken; unclear it's even a business document.     | No discernible argument or structure.                             | Domains irrelevant—text is effectively noise.                                  | Reads like random text generation without intent.                              |

### Output
Respond with a JSON object of the following form:
{
    "coherence_score_reasoning": <str: a single very concise sentence explaining the reason for the score>
    "coherence_score": <int: coherence score>
}
"""
                coh_resp = await openrouter_client.chat.completions.create(
                    model="o4-mini",
                    messages=[
                        {"role": "system", "content": validation_prompt},
                        {"role": "user", "content": content}
                    ],
                    response_format={"type": "json_object"}
                )
                coh_content = coh_resp.choices[0].message.content
                # Parse JSON result
                if isinstance(coh_content, str):
                    coh_json = json.loads(coh_content)
                else:
                    coh_json = coh_content
                
                coherence_score = coh_json.get("coherence_score", 0)
                coherence_reasoning = coh_json.get("coherence_score_reasoning", "")
                
                # Check if coherence score is less than configured minimum
                if coherence_score < min_coherence_score:
                    print(f"Low coherence score ({coherence_score}) for {model_name} (num_constraints={num_constraints}, seed={seed}, attempt {attempt}/{max_retries}): {coherence_reasoning}. Retrying...")
                    await asyncio.sleep(retry_delay)
                    _log_attempt(model_label, seed, num_constraints, attempt, 'rejected', 'coherence_lt6', coherence_score)
                    continue
                
                # Return content with coherence information and API call latency
                _log_attempt(model_label, seed, num_constraints, attempt, 'success', 'ok', coherence_score)
                return {
                    "content": content,
                    "coherence_score": coherence_score,
                    "coherence_score_reasoning": coherence_reasoning,
                    "latency_seconds": api_latency
                }
            except Exception as e:
                print(f"Coherence validation error for {model_name} (num_constraints={num_constraints}, seed={seed}, attempt {attempt}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay)
                _log_attempt(model_label, seed, num_constraints, attempt, 'error', 'validation_error')
                continue
        except Exception as e:
            print(f"Error fetching content from {model_name} (num_constraints={num_constraints}, seed={seed}, attempt {attempt}/{max_retries}): {e}")
            await asyncio.sleep(retry_delay)
            _log_attempt(model_label, seed, num_constraints, attempt, 'error', 'api_error')
    raise RuntimeError(f"Failed to get valid response from model {model_name} after {max_retries} retries")

async def get_completion_async(prompt: str, model_name: str, seed: int = None, num_constraints: int = None, reasoning_effort: str | None = None, run_label: str | None = None, provider=None) -> dict:
    return await call_model(prompt, model_name, seed, num_constraints, reasoning_effort, run_label) 