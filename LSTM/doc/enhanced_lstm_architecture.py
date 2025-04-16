import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
import matplotlib.colors as mcolors
from matplotlib.path import Path
import matplotlib.patches as patches

def create_enhanced_lstm_architecture():
    """
    Create an enhanced LSTM model architecture diagram
    
    This function generates a visualization of the architecture diagram, showing how
    the neural network model combining autoencoder, multi-head attention and LSTM
    processes neuron data
    """
    # 设置图表大小和样式
    plt.figure(figsize=(18, 12))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 创建主画布
    ax = plt.gca()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # 定义颜色方案
    colors = {
        'input': '#3498db',       # 蓝色
        'encoder': '#e74c3c',     # 红色
        'decoder': '#9b59b6',     # 紫色
        'attention': '#f39c12',   # 橙色
        'lstm': '#2ecc71',        # 绿色
        'output': '#1abc9c',      # 青绿色
        'kmeans': '#34495e',      # 深灰蓝色
        'arrow': '#95a5a6',       # 灰色
        'text': '#2c3e50',        # 深蓝色
        'background': '#ecf0f1'   # 浅灰色
    }
    
    # Draw title
    plt.title('Enhanced LSTM Neural Network Architecture', fontsize=24, color=colors['text'], pad=20, fontweight='bold')
    
    # Draw input layer
    draw_box(ax, 10, 75, 15, 10, colors['input'], 'Neuron\nInput Data\n(X_t)', fontsize=10)
    
    # Draw K-means preprocessing part
    draw_box(ax, 10, 60, 15, 10, colors['kmeans'], 'K-means\nClustering', fontsize=10)
    draw_arrow(ax, 17.5, 75, 17.5, 70, colors['arrow'])
    
    # Draw autoencoder part
    draw_box(ax, 35, 75, 20, 20, colors['background'], 'Autoencoder', fontsize=12, alpha=0.3)
    draw_box(ax, 35, 80, 15, 8, colors['encoder'], 'Encoder', fontsize=10)
    draw_box(ax, 35, 68, 15, 8, colors['decoder'], 'Decoder', fontsize=10)
    draw_text(ax, 35, 74, 'Latent Space', fontsize=9, color=colors['text'])
    
    # Draw connection from input to autoencoder
    draw_arrow(ax, 25, 80, 27.5, 80, colors['arrow'])
    
    # Draw connection from K-means to autoencoder
    draw_arrow(ax, 25, 65, 30, 72, colors['arrow'])
    
    # Draw multi-head attention mechanism
    draw_box(ax, 65, 75, 20, 15, colors['attention'], 'Multi-Head Attention', fontsize=10)
    draw_multi_head_attn(ax, 65, 75, colors)
    
    # Connect autoencoder to attention mechanism
    draw_arrow(ax, 50, 80, 55, 80, colors['arrow'])
    
    # Draw LSTM layer
    draw_box(ax, 45, 45, 30, 20, colors['lstm'], 'LSTM Layer\n(Hidden state h_t and cell state c_t)', fontsize=11)
    draw_lstm_details(ax, 45, 45, colors)
    
    # Connect attention mechanism to LSTM
    draw_arrow(ax, 65, 67.5, 65, 55, colors['arrow'])
    
    # Draw temporal attention mechanism
    draw_box(ax, 65, 30, 20, 10, colors['attention'], 'Temporal Attention', fontsize=10)
    
    # Connect LSTM to temporal attention
    draw_arrow(ax, 60, 35, 55, 35, colors['arrow'])
    
    # Draw output layer
    draw_box(ax, 65, 15, 15, 10, colors['output'], 'Output Layer\n(Classification)', fontsize=10)
    
    # Connect temporal attention to output layer
    draw_arrow(ax, 65, 30, 65, 25, colors['arrow'])
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=colors['input'], label='Input Data Layer'),
        patches.Patch(facecolor=colors['encoder'], label='Encoder Component'),
        patches.Patch(facecolor=colors['decoder'], label='Decoder Component'),
        patches.Patch(facecolor=colors['attention'], label='Attention Mechanism'),
        patches.Patch(facecolor=colors['lstm'], label='LSTM Layer'),
        patches.Patch(facecolor=colors['output'], label='Output/Prediction Layer'),
        patches.Patch(facecolor=colors['kmeans'], label='K-means Clustering')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add additional explanation text
    explanation = (
        "Model Explanation:\n"
        "1. Input neuron data is processed through K-means for clustering analysis and feature extraction\n"
        "2. Autoencoder reduces dimensionality and extracts latent features\n"
        "3. Multi-head attention mechanism focuses on different important parts in the sequence\n"
        "4. LSTM layer processes temporal dependencies\n"
        "5. Temporal attention mechanism processes key timepoints in long sequences\n"
        "6. Final output layer generates behavioral prediction results"
    )
    plt.figtext(0.02, 0.02, explanation, fontsize=11, color=colors['text'], 
               bbox=dict(facecolor='white', alpha=0.5, edgecolor=colors['text']))
    
    # Save image
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = 'D:/GitHub_local/Brain_Neuroimage_Processing/LSTM/doc/enhanced_lstm_architecture.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram saved to: {save_path}")
    
    return save_path


def draw_box(ax, x, y, width, height, color, text, fontsize=10, alpha=1.0):
    """Draw a rectangle box with text"""
    rect = plt.Rectangle((x - width/2, y - height/2), width, height, 
                       facecolor=color, alpha=alpha, edgecolor='black', lw=1)
    ax.add_patch(rect)
    draw_text(ax, x, y, text, fontsize)


def draw_text(ax, x, y, text, fontsize=10, color='black'):
    """Draw text at specified position"""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
           color=color, fontweight='bold')


def draw_arrow(ax, x1, y1, x2, y2, color, width=0.5):
    """Draw an arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                          color=color, lw=width, mutation_scale=15)
    ax.add_patch(arrow)


def draw_lstm_details(ax, x, y, colors):
    """Draw simplified representation of LSTM internal structure"""
    # Four gates of LSTM cell
    gates = ['Input', 'Forget', 'Output', 'Cell']
    positions = [(x-10, y+5), (x-10, y-5), (x+10, y+5), (x+10, y-5)]
    
    for gate, pos in zip(gates, positions):
        circle = plt.Circle(pos, 3, facecolor='white', edgecolor=colors['lstm'])
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], gate[0], ha='center', va='center', fontsize=8, color=colors['text'])
    
    # 连接各个门
    for i, start in enumerate(positions):
        for j, end in enumerate(positions):
            if i != j:
                arrow = FancyArrowPatch(start, end, arrowstyle='->', 
                                      connectionstyle="arc3,rad=0.3",
                                      color=colors['arrow'], lw=0.3)
                ax.add_patch(arrow)


def draw_multi_head_attn(ax, x, y, colors):
    """Draw simplified representation of multi-head attention"""
    head_positions = [(x-7, y+3), (x, y+3), (x+7, y+3)]
    output_position = (x, y-4)
    
    for i, pos in enumerate(head_positions):
        circle = plt.Circle(pos, 2, facecolor='white', edgecolor=colors['attention'])
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], f"H{i+1}", ha='center', va='center', fontsize=7, color=colors['text'])
        
        # 连接到输出
        arrow = FancyArrowPatch(pos, output_position, arrowstyle='->', 
                              color=colors['arrow'], lw=0.3)
        ax.add_patch(arrow)
    
    # Output node
    output = plt.Circle(output_position, 2.5, facecolor='white', edgecolor=colors['attention'])
    ax.add_patch(output)
    ax.text(output_position[0], output_position[1], "Concat", ha='center', va='center', fontsize=7, color=colors['text'])


if __name__ == "__main__":
    image_path = create_enhanced_lstm_architecture()
    print(f"Architecture diagram successfully generated: {image_path}")