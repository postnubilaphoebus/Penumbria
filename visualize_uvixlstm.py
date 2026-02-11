import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_uvixlstm_diagram():
    """
    Create a UNet-style visualization of the modified UViXLSTM architecture
    highlighting Zernike layer, FRN normalization, and encoder-decoder structure
    """
    fig, ax = plt.subplots(figsize=(24, 22))
    ax.set_xlim(0, 24)
    ax.set_ylim(-3.5, 20.5)
    ax.axis('off')
    
    # Color scheme
    color_input = '#E8F4F8'
    color_zernike = '#B4E7F5'
    color_encoder = '#7FCDBB'
    color_bottleneck = '#41B6C4'
    color_decoder = '#FDAE6B'
    color_output = '#F16913'
    color_skip = '#666666'
    
    # Helper function to draw a block
    def draw_block(x, y, width, height, color, label, sublabels=None, alpha=0.8):
        box = FancyBboxPatch((x, y), width, height,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=color,
                            linewidth=2, alpha=alpha)
        ax.add_patch(box)
        
        # Calculate vertical spacing based on number of lines
        num_lines = label.count('\n') + 1
        if sublabels:
            num_lines += len(sublabels)
        
        # Adjust main label position - move up if there are sublabels
        if sublabels:
            label_y = y + height * 0.65
        else:
            label_y = y + height/2
            
        ax.text(x + width/2, label_y, label,
               ha='center', va='center', fontsize=28, fontweight='bold')
        
        if sublabels:
            # Start sublabels lower with better spacing
            y_offset = y + height * 0.35
            for i, sublabel in enumerate(sublabels):
                ax.text(x + width/2, y_offset, sublabel,
                       ha='center', va='center', fontsize=22, style='italic')
                y_offset -= height * 0.2  # Increased spacing between sublabels
    
    # Helper function to draw arrow
    def draw_arrow(x1, y1, x2, y2, style='->', color='black', linewidth=2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle=style, color=color,
                               linewidth=linewidth, mutation_scale=40)
        ax.add_patch(arrow)
    
    # Helper function to draw straight skip connection
    def draw_skip_connection(x1, y1, x2, y2):
        # Draw straight horizontal skip connection
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', color=color_skip,
                               linewidth=4, mutation_scale=40, 
                               linestyle='--', alpha=0.7)
        ax.add_patch(arrow)
    
    # Dimensions for blocks - much larger blocks with even more height
    block_width = 2.8
    block_height = 2.6
    horizontal_spacing = 0.5
    vertical_spacing = 2.8
    
    # Starting positions - left side for encoder, shifted up higher
    start_x = 1.5
    start_y = 17.5
    
    # Input
    draw_block(start_x, start_y, block_width, block_height, 
              color_input, 'Input\n128³×1', alpha=0.9)
    current_x = start_x
    current_y = start_y - vertical_spacing
    draw_arrow(start_x + block_width/2, start_y, 
              start_x + block_width/2, current_y + block_height,
              color='black', linewidth=4)
    
    # Zernike Layer
    draw_block(current_x, current_y, block_width, block_height,
              color_zernike, 'Zernike\nLayer',
              sublabels=['j ∈ {3,4,12}', 'Spectral Dropout'])
    
    # Arrow from Zernike to next layer - start from bottom of Zernike
    current_y -= vertical_spacing
    draw_arrow(start_x + block_width/2, current_y + vertical_spacing,
              start_x + block_width/2, current_y + block_height,
              color='black', linewidth=4)
    
    # Encoder path - vertical on the left
    encoder_configs = [
        ('Conv 7×7\n64³×64', 'FRN'),
        ('Bottleneck\n32³×128', 'FRN'),
        ('Bottleneck\n16³×256', 'FRN'),
        ('Bottleneck\n8³×512', 'FRN'),
    ]
    
    encoder_x_positions = []
    encoder_y_positions = []
    
    for i, (main_label, norm_label) in enumerate(encoder_configs):
        draw_block(current_x, current_y, block_width, block_height,
                  color_encoder, main_label, sublabels=[norm_label])
        encoder_x_positions.append(current_x)
        encoder_y_positions.append(current_y)
        
        if i < len(encoder_configs) - 1:
            # Downward arrow
            draw_arrow(current_x + block_width/2, current_y,
                      current_x + block_width/2, current_y - vertical_spacing + block_height,
                      color='#2C7BB6', linewidth=5)
            current_y -= vertical_spacing
    
    # Bottleneck - at the bottom center
    bottleneck_x = start_x + block_width + 4.5
    bottleneck_y = encoder_y_positions[-1]
    
    # Arrow from encoder to bottleneck - meet at middle right of encoder
    draw_arrow(current_x + block_width, encoder_y_positions[-1] + block_height/2,
              bottleneck_x, bottleneck_y + block_height,
              color='#2C7BB6', linewidth=5)
    
    draw_block(bottleneck_x, bottleneck_y, block_width*1.8, block_height*2.0,
              color_bottleneck, 'xLSTM Blocks\n(depth=12)',
              sublabels=['4³×256', 'Bidirectional Vision-LSTM'])
    
    # Decoder path - vertical on the right, going upward
    decoder_x = bottleneck_x + block_width*1.8 + 4.5
    decoder_start_y = encoder_y_positions[-1]  # Align with last encoder block
    
    # Arrow from bottleneck to first decoder - meet at middle left of first decoder
    draw_arrow(bottleneck_x + block_width*1.8, bottleneck_y + block_height,
              decoder_x, decoder_start_y + block_height/2,
              color='#D7301F', linewidth=5)
    
    decoder_configs = [
        ('Decoder\n16³×128', 'FRN'),
        ('Decoder\n32³×64', 'FRN'),
        ('Decoder\n64³×32', 'FRN'),
        ('Decoder\n128³×8', 'FRN'),
    ]
    
    current_y = decoder_start_y
    for i, (main_label, norm_label) in enumerate(decoder_configs):
        draw_block(decoder_x, current_y, block_width, block_height,
                  color_decoder, main_label, sublabels=[norm_label])
        
        # Skip connections from encoder to decoder - straight horizontal lines
        if i < len(decoder_configs):
            enc_idx = len(encoder_configs) - 1 - i
            draw_skip_connection(
                encoder_x_positions[enc_idx] + block_width,
                encoder_y_positions[enc_idx] + block_height/2,
                decoder_x,
                current_y + block_height/2
            )
        
        if i < len(decoder_configs) - 1:
            # Upward arrow
            draw_arrow(decoder_x + block_width/2, current_y + block_height,
                      decoder_x + block_width/2, current_y + vertical_spacing,
                      color='#D7301F', linewidth=5)
            current_y += vertical_spacing
    
    # Output
    output_y = current_y + vertical_spacing
    draw_arrow(decoder_x + block_width/2, current_y + block_height,
              decoder_x + block_width/2, output_y,
              color='black', linewidth=4)
    
    draw_block(decoder_x, output_y, block_width, block_height,
              color_output, 'Output\n128³×C', alpha=0.9)
    
    # Add title at top center
    ax.text(12, 19.0, 'Modified UViXLSTM Architecture',
           ha='center', fontsize=40, fontweight='bold')
    
    # Add legends at bottom with plenty of space
    # Skip connection legend - bottom left
    skip_legend_x = 8.0
    skip_legend_y = -0.5
    ax.text(skip_legend_x, skip_legend_y + 0.7, 'Skip Connections',
           fontsize=28, fontweight='bold', ha='center')
    draw_skip_connection(skip_legend_x - 1.2, skip_legend_y, skip_legend_x + 1.2, skip_legend_y)
    
    # Add legend for modifications - bottom right, aligned
    legend_x = 11.5
    legend_y = 0.2
    
    ax.text(legend_x + 2.5, legend_y, 'Key Modifications:', 
           fontsize=30, fontweight='bold')
    
    modifications = [
        ('• Zernike Phase Layer (j ∈ {3,4,12})', color_zernike),
        ('• Spectral Dropout regularization', color_zernike),
        ('• FRN (Filter Response Norm + TLU)', color_encoder),
    ]
    
    for i, (text, color) in enumerate(modifications):
        y_offset = legend_y - 0.7 - i * 0.8
        # Small colored box
        rect = patches.Rectangle((legend_x, y_offset - 0.3), 0.5, 0.6,
                                linewidth=2, edgecolor='black',
                                facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        ax.text(legend_x + 0.7, y_offset, text, fontsize=26, va='center')
    
    plt.tight_layout()
    return fig

def create_detailed_block_diagram():
    """
    Create a more detailed view of the encoder bottleneck showing FRN usage
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Encoder Bottleneck Block Detail\n(FRN Normalization)',
           ha='center', fontsize=12, fontweight='bold')
    
    # Draw the bottleneck block structure
    operations = [
        ('Input\nC_in', 1, 5.5, '#E8F4F8'),
        ('Conv3D 1×1\nC_in → width', 1, 4.5, '#7FCDBB'),
        ('FRN + TLU', 1, 3.8, '#B4E7F5'),
        ('Conv3D 3×3\nstride=2', 1, 3.0, '#7FCDBB'),
        ('FRN + TLU', 1, 2.3, '#B4E7F5'),
        ('Conv3D 1×1\nwidth → C_out', 1, 1.5, '#7FCDBB'),
        ('FRN', 1, 0.8, '#B4E7F5'),
    ]
    
    # Residual path
    residual_ops = [
        ('Input\nC_in', 7, 5.5, '#E8F4F8'),
        ('Conv3D 1×1\nstride=2', 7, 3.5, '#FDAE6B'),
        ('FRN', 7, 2.5, '#B4E7F5'),
    ]
    
    # Draw main path
    for i, (label, x, y, color) in enumerate(operations):
        rect = FancyBboxPatch((x, y), 1.5, 0.5,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=color,
                             linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + 0.75, y + 0.25, label,
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        if i < len(operations) - 1:
            ax.arrow(x + 0.75, y, 0, -0.15, head_width=0.15,
                    head_length=0.1, fc='black', ec='black')
    
    # Draw residual path
    for i, (label, x, y, color) in enumerate(residual_ops):
        rect = FancyBboxPatch((x, y), 1.5, 0.5,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=color,
                             linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + 0.75, y + 0.25, label,
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        if i < len(residual_ops) - 1:
            ax.arrow(x + 0.75, y, 0, -0.35, head_width=0.15,
                    head_length=0.1, fc='gray', ec='gray', linestyle='--')
    
    # Addition operation
    ax.add_patch(patches.Circle((4.5, 0.5), 0.3, color='white',
                               ec='black', linewidth=2))
    ax.text(4.5, 0.5, '+', ha='center', va='center',
           fontsize=16, fontweight='bold')
    
    # Arrows to addition
    ax.arrow(1.75, 0.55, 2.3, -0.05, head_width=0.15,
            head_length=0.1, fc='black', ec='black')
    ax.arrow(7.75, 2.25, -2.5, -1.5, head_width=0.15,
            head_length=0.1, fc='gray', ec='gray', linestyle='--')
    
    # Output
    rect = FancyBboxPatch((3.75, -0.3), 1.5, 0.5,
                         boxstyle="round,pad=0.05",
                         edgecolor='black', facecolor='#41B6C4',
                         linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(4.5, -0.05, 'Output\nC_out',
           ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add annotations
    ax.text(2, 6.5, 'Main Path', fontsize=10, fontweight='bold', color='#2C7BB6')
    ax.text(7.75, 6.5, 'Downsample Path', fontsize=10, fontweight='bold', color='gray')
    
    ax.text(6, 1.0, 'FRN = Filter Response Normalization\nReplaces BatchNorm + ReLU',
           ha='center', fontsize=8, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # Generate main architecture diagram
    fig1 = create_uvixlstm_diagram()
    fig1.savefig('uvixlstm_architecture.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
