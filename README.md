# IFScale

A novel benchmark for measuring LLM instruction-following performance degradation as instruction density increases.

## Live Site

Visit the benchmark leaderboard at: https://distylai.github.io/IFScale

## About

Production-grade LLM systems require robust adherence to dozens or even hundreds of instructions simultaneously. From content generation systems that must adhere to style guidelines and factual requirements, to automated workflows that integrate dozens of business rules and
compliance standards, to agentic systems requiring robust memory layers and tool usage, modern applications
demand models that can execute complex tasks while satisfying multiple simultaneous instructions.

**IFScale** addresses a critical gap in LLM evaluation: Our benchmark measures how instruction-following performance degrades as instruction density increases up to 500 simultaneous instructions. Our instructions are keyword-inclusion requirements for a business report writing task
of the form  "Include the exact word {keyword}".

## Key Findings

Our evaluation of **20 state-of-the-art models across seven major providers** reveals:

- **3 distinct performance degradation patterns** as instruction density increases
- **Bias for earlier instructions** that only increases until models hit an instruction saturation point 
- **Distinct categories of instruction-following errors** under increased cognitive load
- **Reasoning models** maintain superior performance but exhibit increased variance at extreme instruction densities

## Features

- **Cognitive Load Testing**: Systematic evaluation from single instructions to hundreds of simultaneous requirements
- **Degradation Analysis**: Performance pattern analysis including primacy effects and error type classification
- **Multi-Provider Coverage**: Comprehensive evaluation across seven major AI providers

## Methodology

The benchmark evaluates models across different instruction density levels from 10-500 simultaneous instructions with a step size of 10.
We report accuracy averaged across 5 runs and variance at each density level at 10, 50, 100, 250, and 500 instructions.

Additionally, full results are available at IFScale_results.csv

## Running the Benchmark on Your Own Model

Coming soon!

## Citation

If you use IFScale in your research, please cite:

```bibtex
@misc{jaroslawicz2025instructionsllmsfollowonce,
    title={How Many Instructions Can LLMs Follow at Once?}, 
    author={Daniel Jaroslawicz and Brendan Whiting and Parth Shah and Karime Maamari},
    year={2025},
    eprint={2507.11538},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2507.11538}, 
}
```

## License

MIT License - Open for research and commercial use.
