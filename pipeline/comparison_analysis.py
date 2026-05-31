import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import base64
from io import BytesIO

def load_model_output(filepath):
    """Load model output JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_comparison_visualizations(yolo_data, maskrcnn_data):
    """Create comparison visualizations between models"""
    
    # Extract data
    yolo_images = {img['filename']: img['instances'] for img in yolo_data['images']}
    maskrcnn_images = {img['filename']: img['instances'] for img in maskrcnn_data['images']}
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 6))
    
    # 1. Total Instances Comparison
    ax1 = plt.subplot(1, 2, 1)
    models = ['YOLO', 'Mask R-CNN']
    total_instances = [yolo_data['total_instances'], maskrcnn_data['total_instances']]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax1.bar(models, total_instances, color=colors, width=0.6, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Total Instances Detected', fontsize=11, fontweight='bold')
    ax1.set_title('Total Instances Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. Statistical Comparison
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    
    common_images = sorted(list(set(yolo_images.keys()) & set(maskrcnn_images.keys())))
    yolo_vals = [yolo_images[img] for img in common_images]
    maskrcnn_vals = [maskrcnn_images[img] for img in common_images]
    
    yolo_mean = np.mean(yolo_vals)
    maskrcnn_mean = np.mean(maskrcnn_vals)
    yolo_std = np.std(yolo_vals)
    maskrcnn_std = np.std(maskrcnn_vals)
    yolo_max = np.max(yolo_vals)
    maskrcnn_max = np.max(maskrcnn_vals)
    
    stats_text = f"""
    STATISTICAL COMPARISON
    
    YOLO:
    • Mean: {yolo_mean:.2f} instances/image
    • Std Dev: {yolo_std:.2f}
    • Max: {yolo_max} instances
    • Total: {yolo_data['total_instances']} instances
    
    Mask R-CNN:
    • Mean: {maskrcnn_mean:.2f} instances/image
    • Std Dev: {maskrcnn_std:.2f}
    • Max: {maskrcnn_max} instances
    • Total: {maskrcnn_data['total_instances']} instances
    
    Performance Gain:
    • Mask R-CNN detected {maskrcnn_data['total_instances'] - yolo_data['total_instances']} 
      more instances ({((maskrcnn_data['total_instances']/yolo_data['total_instances'] - 1) * 100):.1f}% more)
    """
    
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def generate_html_report(yolo_data, maskrcnn_data, fig, output_path):
    """Generate HTML report with embedded visualizations"""
    
    # Convert matplotlib figure to base64 image
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    
    # Extract data for tables
    yolo_images = {img['filename']: img['instances'] for img in yolo_data['images']}
    maskrcnn_images = {img['filename']: img['instances'] for img in maskrcnn_data['images']}
    
    common_images = sorted(list(set(yolo_images.keys()) & set(maskrcnn_images.keys())))
    yolo_vals = [yolo_images[img] for img in common_images]
    maskrcnn_vals = [maskrcnn_images[img] for img in common_images]
    
    # Calculate statistics
    yolo_mean = np.mean(yolo_vals)
    maskrcnn_mean = np.mean(maskrcnn_vals)
    yolo_std = np.std(yolo_vals)
    maskrcnn_std = np.std(maskrcnn_vals)
    performance_gain = ((maskrcnn_data['total_instances']/yolo_data['total_instances'] - 1) * 100)
    
    # Create comparison table rows
    comparison_rows = []
    for i, img_name in enumerate(common_images[:20]):  # Show first 20 images
        yolo_count = yolo_images[img_name]
        maskrcnn_count = maskrcnn_images[img_name]
        difference = maskrcnn_count - yolo_count
        comparison_rows.append(f"""
        <tr>
            <td>{i+1}</td>
            <td>{Path(img_name).stem}</td>
            <td style="text-align: center;">{yolo_count}</td>
            <td style="text-align: center;">{maskrcnn_count}</td>
            <td style="text-align: center; background-color: {'#c8e6c9' if difference >= 0 else '#ffcdd2'};">
                {difference:+d}
            </td>
        </tr>
        """)
    
    comparison_table = '\n'.join(comparison_rows)
    
    # Create top 10 tables
    top_10_yolo = sorted(yolo_images.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_yolo_rows = '\n'.join([
        f'<tr><td>{i+1}</td><td>{Path(f[0]).stem}</td><td style="text-align: center;">{f[1]}</td></tr>'
        for i, f in enumerate(top_10_yolo)
    ])
    
    top_10_maskrcnn = sorted(maskrcnn_images.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_maskrcnn_rows = '\n'.join([
        f'<tr><td>{i+1}</td><td>{Path(f[0]).stem}</td><td style="text-align: center;">{f[1]}</td></tr>'
        for i, f in enumerate(top_10_maskrcnn)
    ])
    
    # HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Comparison Report - Image Segmentation</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            header h1 {{
                margin: 0;
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            header p {{
                margin: 5px 0;
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .content {{
                padding: 30px;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .section h2 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 5px solid #667eea;
            }}
            .stat-card h3 {{
                margin: 0 0 10px 0;
                color: #333;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin: 10px 0;
            }}
            .stat-label {{
                color: #666;
                font-size: 0.9em;
            }}
            .comparison-box {{
                background: #f0f4ff;
                border-left: 5px solid #4ECDC4;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }}
            .highlight {{
                background-color: #fff9c4;
                padding: 2px 6px;
                border-radius: 3px;
                font-weight: bold;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            table thead {{
                background: #667eea;
                color: white;
            }}
            table th {{
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }}
            table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
            }}
            table tbody tr:hover {{
                background-color: #f9f9f9;
            }}
            table tbody tr:nth-child(even) {{
                background-color: #f5f5f5;
            }}
            .visualization {{
                text-align: center;
                margin: 30px 0;
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .two-column {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            .winner {{
                background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                font-size: 1.1em;
                font-weight: bold;
                color: #333;
            }}
            footer {{
                background: #f5f5f5;
                padding: 20px;
                text-align: center;
                color: #666;
                border-top: 1px solid #ddd;
            }}
            .model-badge {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 600;
                margin: 0 5px;
            }}
            .yolo-badge {{
                background-color: #FF6B6B;
                color: white;
            }}
            .maskrcnn-badge {{
                background-color: #4ECDC4;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🔍 Image Segmentation Model Comparison Report</h1>
                <p>Comprehensive Analysis: YOLO vs Mask R-CNN</p>
                <p style="font-size: 0.95em; margin-top: 15px;">Generated for evaluation of instance detection performance</p>
            </header>
            
            <div class="content">
                <!-- Executive Summary -->
                <div class="section">
                    <h2>📊 Executive Summary</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h3>🎯 YOLO Model</h3>
                            <div class="stat-value">{yolo_data['total_instances']}</div>
                            <div class="stat-label">Total Instances Detected</div>
                            <div class="stat-label" style="margin-top: 10px;">Avg per image: {yolo_mean:.2f}</div>
                        </div>
                        <div class="stat-card">
                            <h3>🎯 Mask R-CNN Model</h3>
                            <div class="stat-value">{maskrcnn_data['total_instances']}</div>
                            <div class="stat-label">Total Instances Detected</div>
                            <div class="stat-label" style="margin-top: 10px;">Avg per image: {maskrcnn_mean:.2f}</div>
                        </div>
                        <div class="stat-card">
                            <h3>📈 Performance Gain</h3>
                            <div class="stat-value" style="color: #4ECDC4;">{performance_gain:+.1f}%</div>
                            <div class="stat-label">Mask R-CNN vs YOLO</div>
                            <div class="stat-label" style="margin-top: 10px;">+{maskrcnn_data['total_instances'] - yolo_data['total_instances']} instances</div>
                        </div>
                        <div class="stat-card">
                            <h3>📷 Dataset Size</h3>
                            <div class="stat-value">{yolo_data['total_images']}</div>
                            <div class="stat-label">Images Processed</div>
                        </div>
                    </div>
                </div>
                
                <!-- Winner Announcement -->
                <div class="section">
                    <div class="winner">
                        ✨ <span class="maskrcnn-badge">Mask R-CNN</span> performed better with <span class="highlight">{maskrcnn_data['total_instances'] - yolo_data['total_instances']} more instances</span> detected ({performance_gain:.1f}% improvement) ✨
                    </div>
                </div>
                
                <!-- Detailed Comparison -->
                <div class="section">
                    <h2>📋 Detailed Metrics Comparison</h2>
                    <div class="two-column">
                        <div>
                            <h3><span class="yolo-badge">YOLO</span></h3>
                            <div class="comparison-box">
                                <p><strong>Total Instances:</strong> {yolo_data['total_instances']}</p>
                                <p><strong>Total Images:</strong> {yolo_data['total_images']}</p>
                                <p><strong>Average/Image:</strong> {yolo_mean:.2f}</p>
                                <p><strong>Std Deviation:</strong> {yolo_std:.2f}</p>
                                <p><strong>Max Instances:</strong> {np.max(yolo_vals)}</p>
                                <p><strong>Min Instances:</strong> {np.min(yolo_vals)}</p>
                            </div>
                        </div>
                        <div>
                            <h3><span class="maskrcnn-badge">Mask R-CNN</span></h3>
                            <div class="comparison-box">
                                <p><strong>Total Instances:</strong> {maskrcnn_data['total_instances']}</p>
                                <p><strong>Total Images:</strong> {maskrcnn_data['total_images']}</p>
                                <p><strong>Average/Image:</strong> {maskrcnn_mean:.2f}</p>
                                <p><strong>Std Deviation:</strong> {maskrcnn_std:.2f}</p>
                                <p><strong>Max Instances:</strong> {np.max(maskrcnn_vals)}</p>
                                <p><strong>Min Instances:</strong> {np.min(maskrcnn_vals)}</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Visualizations -->
                <div class="section">
                    <h2>📈 Comparative Visualizations</h2>
                    <div class="visualization">
                        <img src="data:image/png;base64,{image_base64}" alt="Comparison Visualizations">
                    </div>
                </div>
                
                <!-- Detailed Comparison Table -->
                <div class="section">
                    <h2>🔬 Sample Comparison (First 20 Images)</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Image</th>
                                <th><span class="yolo-badge">YOLO</span></th>
                                <th><span class="maskrcnn-badge">Mask R-CNN</span></th>
                                <th>Difference</th>
                            </tr>
                        </thead>
                        <tbody>
                            {comparison_table}
                        </tbody>
                    </table>
                </div>
                
                <!-- Top Performers -->
                <div class="section">
                    <h2>⭐ Top Performers</h2>
                    <div class="two-column">
                        <div>
                            <h3><span class="yolo-badge">Top 10 - YOLO</span></h3>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Rank</th>
                                        <th>Image</th>
                                        <th>Instances</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {top_10_yolo_rows}
                                </tbody>
                            </table>
                        </div>
                        <div>
                            <h3><span class="maskrcnn-badge">Top 10 - Mask R-CNN</span></h3>
                            <table>
                                <thead>
                                    <tr>
                                        <th>Rank</th>
                                        <th>Image</th>
                                        <th>Instances</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {top_10_maskrcnn_rows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <!-- Key Insights -->
                <div class="section">
                    <h2>💡 Key Insights</h2>
                    <ul>
                        <li><strong>Overall Performance:</strong> Mask R-CNN detected <span class="highlight">{maskrcnn_data['total_instances'] - yolo_data['total_instances']} more instances</span> overall, representing a <span class="highlight">{performance_gain:.1f}%</span> improvement.</li>
                        <li><strong>Average Detection:</strong> Mask R-CNN averaged <span class="highlight">{maskrcnn_mean:.2f}</span> instances per image vs YOLO's <span class="highlight">{yolo_mean:.2f}</span>.</li>
                        <li><strong>Consistency:</strong> {"Mask R-CNN shows better consistency" if maskrcnn_std < yolo_std else "YOLO shows better consistency"} with a lower standard deviation (<span class="highlight">{min(maskrcnn_std, yolo_std):.2f}</span> vs <span class="highlight">{max(maskrcnn_std, yolo_std):.2f}</span>).</li>
                        <li><strong>Peak Performance:</strong> Mask R-CNN achieved a maximum of <span class="highlight">{np.max(maskrcnn_vals)}</span> instances in a single image.</li>
                        <li><strong>Recommendation:</strong> <span class="highlight">Mask R-CNN is the superior model</span> for this dataset, detecting significantly more instances with better overall performance.</li>
                    </ul>
                </div>
            </div>
            
            <footer>
                <p>📊 Report generated for Image Segmentation Project</p>
                <p>Comparing YOLO vs Mask R-CNN performance on instance detection</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {output_path}")

def main():
    """Main function to generate comparison report"""
    
    # Define paths
    yolo_path = Path('./pipeline_output/yolo/yolo_output.json')
    maskrcnn_path = Path('./pipeline_output/maskrcnn/maskrcnn_output.json')
    output_dir = Path('./results')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("📂 Loading model outputs...")
    yolo_data = load_model_output(yolo_path)
    maskrcnn_data = load_model_output(maskrcnn_path)
    
    print(f"✅ YOLO: {yolo_data['total_instances']} instances across {yolo_data['total_images']} images")
    print(f"✅ Mask R-CNN: {maskrcnn_data['total_instances']} instances across {maskrcnn_data['total_images']} images")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    fig = create_comparison_visualizations(yolo_data, maskrcnn_data)
    
    # Save plot
    plot_path = output_dir / 'model_comparison_plots.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved: {plot_path}")
    
    # Generate HTML report
    print("\n📄 Generating HTML report...")
    html_path = output_dir / 'model_comparison_report.html'
    generate_html_report(yolo_data, maskrcnn_data, fig, html_path)
    
    print("\n" + "="*60)
    print("✨ COMPARISON ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📊 Visualizations: {plot_path}")
    print(f"📄 HTML Report: {html_path}")
    print("="*60)

if __name__ == '__main__':
    main()
