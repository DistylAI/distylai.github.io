# Density-Bench

A novel benchmark for measuring instruction-following performance degradation as instruction density increases in large language models.

## 🌐 Live Site

Visit the benchmark leaderboard at: https://distyl.github.io

## 📋 About

Recent advances in large language models have expanded context windows from thousands to hundreds of thousands to millions of tokens while dramatically improving reasoning over extended contexts. This progress enables complex single-call requests with dozens or hundreds of simultaneous instructions, scenarios previously requiring decomposition or deemed infeasible.

**Density-Bench** addresses a critical gap in LLM evaluation: while existing benchmarks evaluate models on tasks with single or few instructions, real-world applications often demand adherence to dozens, if not hundreds, of simultaneous requirements. Our benchmark measures how instruction-following performance degrades as instruction density increases.

## 🔬 Key Findings

Our evaluation of **20 state-of-the-art models across seven major providers** reveals:

- **Systematic performance degradation patterns** as instruction density increases
- **Universal recency effects** across all tested models  
- **Error type shifts** under increased cognitive load
- **Reasoning models** maintain superior performance but exhibit increased variance at extreme instruction densities

## 📊 Features

- **🧠 Cognitive Load Testing**: Systematic evaluation from single instructions to hundreds of simultaneous requirements
- **📈 Degradation Analysis**: Performance pattern analysis including recency effects and error type classification
- **🔍 Multi-Provider Coverage**: Comprehensive evaluation across seven major AI providers
- **📱 Interactive Leaderboard**: Real-time performance comparisons with density-specific metrics

## 🚀 Methodology

The benchmark evaluates models across different instruction density levels:

- **Low Density**: 1-5 simultaneous instructions
- **Medium Density**: 10-50 simultaneous instructions  
- **High Density**: 100+ simultaneous instructions

Each density level tests the model's ability to:
- Maintain instruction adherence under cognitive load
- Preserve performance across instruction sequence positions
- Handle error propagation and recovery

## 📈 Metrics

- **Overall Score**: Weighted performance across all density levels
- **Density-Specific Scores**: Performance at low, medium, and high instruction densities
- **Variance**: Consistency measure across different instruction densities
- **Degradation Rate**: Performance drop as density increases

## 🛠️ Development

This site is built as a static HTML page optimized for GitHub Pages:

1. Edit `index.html` to modify leaderboard data or content
2. Push changes to the main branch
3. GitHub Pages automatically deploys updates

## 📊 Adding Results

To add new benchmark results:

1. Update the leaderboard table in `index.html`
2. Include scores for all density levels (Low, Medium, High)
3. Calculate overall score and variance metrics
4. Maintain proper ranking order

## 🎨 Customization

Key sections for customization:

- **Model Data**: Update leaderboard table with latest results
- **Methodology**: Modify density levels or evaluation criteria
- **Metrics**: Add or adjust performance measurement categories
- **Styling**: Customize colors and layout in embedded CSS

## 📚 Citation

If you use Density-Bench in your research, please cite:

```bibtex
@misc{density-bench-2024,
  title={Density-Bench: Measuring Instruction Following Performance Under Cognitive Load},
  year={2024},
  url={https://distyl.github.io}
}
```

## 📝 License

MIT License - Open for research and commercial use.
