import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

def read_and_aggregate_latency():
    """
    Read raw result data from IFScale_results.csv and calculate aggregated latency 
    for each model at different number of rules.
    """
    # Read the CSV file
    df = pd.read_csv('../IFScale_results.csv')
    
    # Group by model and num_rules to calculate aggregated latency statistics
    latency_stats = df.groupby(['model', 'num_rules'])['latency_seconds'].agg([
        'mean',
        'std',
        'median',
        'min',
        'max',
        'count'
    ]).reset_index()
    
    # Rename columns for clarity
    latency_stats.columns = ['model', 'num_rules', 'mean_latency', 'std_latency', 
                            'median_latency', 'min_latency', 'max_latency', 'count']
    
    # Round to reasonable precision
    numeric_cols = ['mean_latency', 'std_latency', 'median_latency', 'min_latency', 'max_latency']
    latency_stats[numeric_cols] = latency_stats[numeric_cols].round(3)
    
    return latency_stats

def get_unique_models_and_rules():
    """
    Get unique models and rule counts from the dataset.
    """
    df = pd.read_csv('../IFScale_results.csv')
    unique_models = sorted(df['model'].unique())
    unique_rules = sorted(df['num_rules'].unique())
    
    print(f"Unique models ({len(unique_models)}):")
    for model in unique_models:
        print(f"  - {model}")
    
    print(f"\nUnique rule counts: {unique_rules}")
    
    return unique_models, unique_rules

def create_latency_pivot_table():
    """
    Create a pivot table showing mean latency for each model at different rule counts.
    """
    latency_stats = read_and_aggregate_latency()
    
    # Create pivot table with models as rows and num_rules as columns
    pivot_table = latency_stats.pivot(index='model', columns='num_rules', values='mean_latency')
    
    return pivot_table

def create_model_mapping():
    """
    Create mapping between model names in different files.
    """
    mapping = {
        'claude-3.5-haiku': 'anthropic/claude-3.5-haiku',
        'claude-3.7-sonnet': 'anthropic/claude-3.7-sonnet', 
        'claude-opus-4': 'anthropic/claude-opus-4',
        'claude-opus-4 (r)': 'anthropic/claude-opus-4-high-reasoning',
        'claude-sonnet-4': 'anthropic/claude-sonnet-4',
        'claude-sonnet-4 (r)': 'anthropic/claude-sonnet-4-high-reasoning',
        'deepseek-r1-0528': 'deepseek/deepseek-r1-0528',
        'gemini-2.5-flash-preview': 'google/gemini-2.5-flash-preview-05-20',
        'gemini-2.5-pro-preview': 'google/gemini-2.5-pro-preview',
        'llama-4-maverick': 'meta-llama/llama-4-maverick',
        'llama-4-scout': 'meta-llama/llama-4-scout',
        'gpt-4.1': 'openai/gpt-4.1',
        'gpt-4.1-mini': 'openai/gpt-4.1-mini',
        'gpt-4.1-nano': 'openai/gpt-4.1-nano',
        'gpt-4.5-preview': 'openai/gpt-4.5-preview',
        'gpt-4o': 'openai/gpt-4o',
        'gpt-4o-mini': 'openai/gpt-4o-mini',
        'o3 (medium)': 'openai/o3',
        'o3 (high)': 'openai/o3-high-reasoning',
        'o4-mini (medium)': 'openai/o4-mini',
        'qwen3-235b-a22b': 'qwen/qwen3-235b-a22b',
        'grok-3-beta': 'x-ai/grok-3-beta',
        'grok-3-mini-beta': 'x-ai/grok-3-mini-beta'
    }
    return mapping

def read_accuracy_data():
    """
    Read accuracy data from results.csv and reshape it into a long format.
    """
    df = pd.read_csv('results.csv')
    model_mapping = create_model_mapping()
    
    # Define the instruction columns and their corresponding rule counts
    instruction_cols = ['10 Instructions accuracy', '50 Instructions accuracy', 
                       '100 Instructions accuracy', '250 Instructions accuracy', 
                       '500 Instructions accuracy']
    rule_counts = [10, 50, 100, 250, 500]
    
    # Create a list to store reshaped data
    accuracy_data = []
    
    for i, col in enumerate(instruction_cols):
        for idx, row in df.iterrows():
            model_short = row['Model']
            # Map to full model name if available
            model_full = model_mapping.get(model_short, model_short)
            accuracy = float(row[col].replace('%', ''))
            num_rules = rule_counts[i]
            
            accuracy_data.append({
                'model': model_full,
                'num_rules': num_rules,
                'accuracy': accuracy
            })
    
    return pd.DataFrame(accuracy_data)

def merge_accuracy_and_latency():
    """
    Merge accuracy and latency data for 3D visualization.
    """
    # Get accuracy data
    accuracy_df = read_accuracy_data()
    
    # Get aggregated latency data
    latency_df = read_and_aggregate_latency()
    
    # Merge on model and num_rules
    merged_df = pd.merge(accuracy_df, latency_df[['model', 'num_rules', 'mean_latency']], 
                        on=['model', 'num_rules'], how='inner')
    
    return merged_df

def create_3d_visualization():
    """
    Create a 3D visualization with number of rules, accuracy, and latency.
    """
    # Get merged data
    df = merge_accuracy_and_latency()
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique models for color mapping
    unique_models = df['model'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_models)))
    
    # Plot each model with different colors
    for i, model in enumerate(unique_models):
        model_data = df[df['model'] == model]
        ax.scatter(model_data['num_rules'], model_data['accuracy'], model_data['mean_latency'],
                  c=[colors[i]], label=model.split('/')[-1], s=60, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Number of Rules', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_zlabel('Mean Latency (seconds)', fontsize=12)
    ax.set_title('3D Visualization: Rules vs Accuracy vs Latency', fontsize=14)
    
    # Add legend (but limit it to avoid overcrowding)
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 10:
        # Show only first 10 models in legend to avoid overcrowding
        ax.legend(handles[:10], labels[:10], bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('3d_rules_accuracy_latency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def create_interactive_plots():
    """
    Create additional 2D plots to complement the 3D visualization.
    """
    df = merge_accuracy_and_latency()
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Accuracy vs Number of Rules
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax1.plot(model_data['num_rules'], model_data['accuracy'], 
                marker='o', alpha=0.7, label=model.split('/')[-1])
    ax1.set_xlabel('Number of Rules')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy vs Number of Rules')
    ax1.grid(True, alpha=0.3)
    
    # 2. Latency vs Number of Rules
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax2.plot(model_data['num_rules'], model_data['mean_latency'], 
                marker='s', alpha=0.7, label=model.split('/')[-1])
    ax2.set_xlabel('Number of Rules')
    ax2.set_ylabel('Mean Latency (seconds)')
    ax2.set_title('Latency vs Number of Rules')
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy vs Latency
    scatter = ax3.scatter(df['accuracy'], df['mean_latency'], 
                         c=df['num_rules'], cmap='viridis', alpha=0.7, s=60)
    ax3.set_xlabel('Accuracy (%)')
    ax3.set_ylabel('Mean Latency (seconds)')
    ax3.set_title('Accuracy vs Latency (colored by Rules)')
    plt.colorbar(scatter, ax=ax3, label='Number of Rules')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics by rule count
    rule_stats = df.groupby('num_rules').agg({
        'accuracy': ['mean', 'std'],
        'mean_latency': ['mean', 'std']
    }).round(2)
    
    rule_means = df.groupby('num_rules')['accuracy'].mean()
    ax4.bar(rule_means.index, rule_means.values, alpha=0.7, color='skyblue')
    ax4.set_xlabel('Number of Rules')
    ax4.set_ylabel('Mean Accuracy (%)')
    ax4.set_title('Average Accuracy by Rule Count')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2d_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return rule_stats

def create_interactive_3d_visualization():
    """
    Create an interactive 3D visualization with plotly showing model performance lines.
    Users can click legend items to show/hide specific models.
    """
    # Get merged data
    df = merge_accuracy_and_latency()
    
    # Create the interactive 3D plot
    fig = go.Figure()
    
    # Get unique models and assign colors
    unique_models = df['model'].unique()
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel
    
    # Add each model as a separate trace (line)
    for i, model in enumerate(unique_models):
        model_data = df[df['model'] == model].sort_values('num_rules')
        
        # Clean model name for display
        model_display = model.split('/')[-1]
        
        fig.add_trace(go.Scatter3d(
            x=model_data['num_rules'],
            y=model_data['accuracy'],
            z=model_data['mean_latency'],
            mode='lines+markers',
            name=model_display,
            line=dict(color=colors[i % len(colors)], width=4),
            marker=dict(size=6, opacity=0.8),
            hovertemplate=
                '<b>%{fullData.name}</b><br>' +
                'Rules: %{x}<br>' +
                'Accuracy: %{y:.1f}%<br>' +
                'Latency: %{z:.2f}s<br>' +
                '<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive 3D Model Performance Analysis<br><sub>Click legend items to show/hide models</sub>',
            'x': 0.5,
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title='Number of Rules',
            yaxis_title='Accuracy (%)',
            zaxis_title='Mean Latency (seconds)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="lightgray",
                showbackground=True,
                zerolinecolor="lightgray",
            ),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="lightgray",
                showbackground=True,
                zerolinecolor="lightgray",
            ),
            zaxis=dict(
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="lightgray",
                showbackground=True,
                zerolinecolor="lightgray",
            )
        ),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        margin=dict(l=0, r=150, t=80, b=0),
        width=1200,
        height=800
    )
    
    # Save as HTML
    fig.write_html("interactive_3d_model_performance.html")
    
    return fig

def create_interactive_dashboard():
    """
    Create a comprehensive interactive dashboard with multiple views.
    """
    df = merge_accuracy_and_latency()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy vs Rules', 'Latency vs Rules', 
                       'Accuracy vs Latency', 'Model Performance Summary'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "table"}]]
    )
    
    # Get unique models and colors
    unique_models = df['model'].unique()
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1
    
    # Plot 1: Accuracy vs Rules
    for i, model in enumerate(unique_models):
        model_data = df[df['model'] == model].sort_values('num_rules')
        model_display = model.split('/')[-1]
        
        fig.add_trace(
            go.Scatter(
                x=model_data['num_rules'],
                y=model_data['accuracy'],
                mode='lines+markers',
                name=model_display,
                line=dict(color=colors[i % len(colors)]),
                showlegend=True,
                legendgroup=model_display
            ),
            row=1, col=1
        )
    
    # Plot 2: Latency vs Rules
    for i, model in enumerate(unique_models):
        model_data = df[df['model'] == model].sort_values('num_rules')
        model_display = model.split('/')[-1]
        
        fig.add_trace(
            go.Scatter(
                x=model_data['num_rules'],
                y=model_data['mean_latency'],
                mode='lines+markers',
                name=model_display,
                line=dict(color=colors[i % len(colors)]),
                showlegend=False,
                legendgroup=model_display
            ),
            row=1, col=2
        )
    
    # Plot 3: Accuracy vs Latency (colored by rules)
    fig.add_trace(
        go.Scatter(
            x=df['accuracy'],
            y=df['mean_latency'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['num_rules'],
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Rules", x=0.45)
            ),
            text=df['model'].str.split('/').str[-1],
            hovertemplate='<b>%{text}</b><br>Accuracy: %{x:.1f}%<br>Latency: %{y:.2f}s<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Plot 4: Summary table
    summary_stats = df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'mean_latency': ['mean', 'std']
    }).round(2)
    
    summary_stats.columns = ['Avg_Accuracy', 'Std_Accuracy', 'Avg_Latency', 'Std_Latency']
    summary_stats = summary_stats.reset_index()
    summary_stats['model'] = summary_stats['model'].str.split('/').str[-1]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Model', 'Avg Accuracy', 'Std Accuracy', 'Avg Latency', 'Std Latency'],
                       fill_color='lightblue',
                       align='left'),
            cells=dict(values=[summary_stats['model'], 
                              summary_stats['Avg_Accuracy'],
                              summary_stats['Std_Accuracy'],
                              summary_stats['Avg_Latency'],
                              summary_stats['Std_Latency']],
                      fill_color='white',
                      align='left')
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Interactive Model Performance Dashboard',
        height=800,
        showlegend=True,
        legend=dict(x=1.02, y=1)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Number of Rules", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Rules", row=1, col=2)
    fig.update_yaxes(title_text="Mean Latency (s)", row=1, col=2)
    fig.update_xaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_yaxes(title_text="Mean Latency (s)", row=2, col=1)
    
    # Save as HTML
    fig.write_html("interactive_dashboard.html")
    
    return fig

def create_model_comparison_tool():
    """
    Create an interactive model comparison tool with dropdown selection.
    """
    df = merge_accuracy_and_latency()
    unique_models = df['model'].unique()
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for all models (initially visible)
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel
    
    for i, model in enumerate(unique_models):
        model_data = df[df['model'] == model].sort_values('num_rules')
        model_display = model.split('/')[-1]
        
        fig.add_trace(go.Scatter3d(
            x=model_data['num_rules'],
            y=model_data['accuracy'],
            z=model_data['mean_latency'],
            mode='lines+markers',
            name=model_display,
            line=dict(color=colors[i % len(colors)], width=4),
            marker=dict(size=6),
            visible=True,
            hovertemplate=
                '<b>%{fullData.name}</b><br>' +
                'Rules: %{x}<br>' +
                'Accuracy: %{y:.1f}%<br>' +
                'Latency: %{z:.2f}s<br>' +
                '<extra></extra>'
        ))
    
    # Create dropdown menus for model selection
    buttons = []
    
    # Show all models
    buttons.append(dict(
        label="All Models",
        method="update",
        args=[{"visible": [True] * len(unique_models)}]
    ))
    
    # Individual model buttons
    for i, model in enumerate(unique_models):
        model_display = model.split('/')[-1]
        visibility = [False] * len(unique_models)
        visibility[i] = True
        
        buttons.append(dict(
            label=model_display,
            method="update",
            args=[{"visible": visibility}]
        ))
    
    # Group buttons by provider
    providers = ['Anthropic', 'OpenAI', 'Google', 'Meta', 'DeepSeek', 'xAI', 'Qwen']
    for provider in providers:
        visibility = []
        for model in unique_models:
            if provider.lower() in model.lower():
                visibility.append(True)
            else:
                visibility.append(False)
        
        if any(visibility):  # Only add if there are models from this provider
            buttons.append(dict(
                label=f"{provider} Models",
                method="update", 
                args=[{"visible": visibility}]
            ))
    
    # Update layout with dropdown
    fig.update_layout(
        title='Interactive 3D Model Comparison Tool',
        scene=dict(
            xaxis_title='Number of Rules',
            yaxis_title='Accuracy (%)',
            zaxis_title='Mean Latency (seconds)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.02,
                yanchor="top"
            ),
        ],
        annotations=[
            dict(text="Select Models:", x=0, xref="paper", y=1.08, yref="paper",
                 align="left", showarrow=False, font=dict(size=14))
        ],
        width=1200,
        height=800,
        margin=dict(t=100)
    )
    
    # Save as HTML
    fig.write_html("model_comparison_tool.html")
    
    return fig

if __name__ == "__main__":
    # Get overview of the data
    print("=== Dataset Overview ===")
    models, rules = get_unique_models_and_rules()
    
    print("\n=== Aggregated Latency Statistics ===")
    latency_stats = read_and_aggregate_latency()
    print(latency_stats.head(20))
    
    print("\n=== Mean Latency Pivot Table ===")
    pivot = create_latency_pivot_table()
    print(pivot)
    
    print("\n=== Reading Accuracy Data ===")
    accuracy_data = read_accuracy_data()
    print(f"Accuracy data shape: {accuracy_data.shape}")
    print(accuracy_data.head(10))
    
    print("\n=== Merging Accuracy and Latency Data ===")
    merged_data = merge_accuracy_and_latency()
    print(f"Merged data shape: {merged_data.shape}")
    print(merged_data.head(10))
    
    print("\n=== Creating 3D Visualization ===")
    plot_data = create_3d_visualization()
    
    print("\n=== Creating Additional 2D Plots ===")
    rule_stats = create_interactive_plots()
    print("\nSummary statistics by rule count:")
    print(rule_stats)
    
    print("\n=== Creating Interactive 3D Visualization ===")
    interactive_3d_fig = create_interactive_3d_visualization()
    print("Created: interactive_3d_model_performance.html")
    
    print("\n=== Creating Interactive Dashboard ===")
    dashboard_fig = create_interactive_dashboard()
    print("Created: interactive_dashboard.html")
    
    print("\n=== Creating Model Comparison Tool ===")
    comparison_fig = create_model_comparison_tool()
    print("Created: model_comparison_tool.html")
    
    # Save the aggregated data
    latency_stats.to_csv('aggregated_latency_stats.csv', index=False)
    pivot.to_csv('latency_pivot_table.csv')
    merged_data.to_csv('merged_accuracy_latency_data.csv', index=False)
    
    print(f"\nSaved data to:")
    print("- aggregated_latency_stats.csv")
    print("- latency_pivot_table.csv") 
    print("- merged_accuracy_latency_data.csv")
    print("- 3d_rules_accuracy_latency.png")
    print("- 2d_analysis_plots.png")
    print("- interactive_3d_model_performance.html")
    print("- interactive_dashboard.html")
    print("- model_comparison_tool.html")