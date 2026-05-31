# Image Segmentation Pipeline

A complete Python pipeline for instance segmentation of images using **Mask R-CNN (ResNet50-FPN)**. Automatically detects and segments individual objects in images with colored instance visualization.

## Features

- **Preprocessing**: Automatically removes blank or corrupt images
- **Instance Segmentation**: Detects and segments individual object instances using Mask R-CNN
- **Colored Output**: All detected instances overlaid with different colors on a single output image
- **JSON Labels**: Per-image metadata including instance details, bounding boxes, and confidence scores
- **Batch Processing**: Process single images or entire folders

## Requirements

- Python 3.8+
- PyTorch and TorchVision
- Pillow, NumPy

## Installation

### 1. Clone or navigate to the repository:

```bash
cd Image-Segmentation
```

### 2. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Note**: If you need CPU-only PyTorch, visit https://pytorch.org for specific installation commands.

## Quick Start

Run the pipeline on your images:

```bash
python pipeline/pipeline.py --input <image-or-folder> --results-dir results
```

## Usage

### Basic Command

```bash
python pipeline/pipeline.py --input <IMAGE_PATH> --results-dir <RESULTS_DIR>
```

### Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|----------|
| `--input` | Yes | Path to a single image or folder | N/A |
| `--workdir` | No | Working directory for intermediate outputs | `pipeline_output` |
| `--results-dir` | No | Final results directory | `results` |
| `--model` | No | Segmentation model to run (`maskrcnn` or `yolo`) | `maskrcnn` |
| `--yolo-weights` | No | Optional YOLO weight file path | None |
| `--groundtruth-labels` | No | Folder containing YOLO-format groundtruth `.txt` labels | None |

### Examples

**Process a single image with Mask R-CNN:**

```bash
python pipeline/pipeline.py --input ./image_data/sample.jpg --model maskrcnn --results-dir results
```

**Process a folder of images with YOLO:**

```bash
python pipeline/pipeline.py --input ./image_data --model yolo --results-dir results
```

**Process with a specific YOLO weight file:**

```bash
python pipeline/pipeline.py --input ./image_data --model yolo --yolo-weights yolov8n-seg.pt --results-dir results
```

**Process and evaluate against groundtruth labels:**

```bash
python pipeline/pipeline.py --input ./image_data --model yolo --results-dir results --groundtruth-labels labels/train2017
```

**Evaluate existing predictions without re-running segmentation:**

```bash
python pipeline/evaluation_report.py --predicted-labels results/yolo/labels --groundtruth-labels labels/train2017 --images image_data --output-dir results/yolo --model-name yolo
```

**Generate struggle/observation report from an existing evaluation:**

```bash
python pipeline/observation_report.py --evaluation results/yolo/evaluation.json --output-dir results/yolo --predicted-labels results/yolo/labels --groundtruth-labels labels/train2017
```

Produces `observation_report.json` and `observation_report.html` highlighting low IoU, false negatives, classification errors, and images with no segmentation.

**Cluster issue images (originals in `issue_eval/images` only — ResNet + K-Means + t-SNE):**

```bash
python pipeline/issue_eval_analysis.py --images-dir issue_eval/images --output-dir issue_eval/reports --groundtruth-labels labels/train2017 --predicted-labels results/yolo/labels
```

Uses COCO 80-class names (YOLOv8-seg). Outputs `issue_eval/reports/issue_cluster_analysis.json`, `.html`, and `tsne_clusters.png`.

## Output Structure

The pipeline creates the following directory structure under the final results directory:

```
results/
└── <model>/
    ├── images/
    │   ├── sample_segmented.png      # All instances colored
    │   ├── photo_segmented.png       # All instances colored
    │   └── ...
    ├── labels/
    │   ├── sample.json               # Instance metadata
    │   ├── photo.json                # Instance metadata
    │   └── ...
    ├── evaluation.json              # Optional evaluation report
    └── output.json                   # Summary results
```

### Output Files Explained

#### Segmented Images (`images/`)

- **`{image}_segmented.png`**: Original image with all detected instances colored differently
  - Each instance is assigned a unique color (Red, Blue, Cyan, Green, Yellow, etc.)
  - Perfect for visualization and quality control
  - Ready for presentation or further analysis

**Example output like your reference image:**
- Red animals, blue animals, cyan animals, etc. all in one image

#### Label Files (`labels/`)

Each `{image}.json` contains instance-level metadata:

```json
{
  "image_file": "sample.jpg",
  "image_size": [1024, 768],
  "instances": [
    {
      "instance_id": 1,
      "confidence": 0.95,
      "bbox": [100, 150, 300, 400],
      "label_id": 1
    },
    {
      "instance_id": 2,
      "confidence": 0.92,
      "bbox": [350, 200, 500, 450],
      "label_id": 1
    }
  ]
}
```

**Fields:**
- `instance_id`: Sequential ID of the detected instance
- `confidence`: Detection confidence score (0-1)
- `bbox`: Bounding box coordinates [x1, y1, x2, y2]
- `label_id`: COCO class ID of the detected object

#### Summary Results (`output.json`)

Overall pipeline results:

```json
{
  "model": "Mask R-CNN (ResNet50-FPN)",
  "total_images": 5,
  "total_instances": 42,
  "images": [
    {"filename": "sample.jpg", "instances": 15},
    {"filename": "photo.jpg", "instances": 10},
    ...
  ]
}
```

## Supported Image Formats

- ✓ JPEG / JPG
- ✓ PNG
- ✓ BMP
- ✓ TIFF
- ✓ WEBP

## Model Information

| Property | Value |
|----------|-------|
| **Model Name** | Mask R-CNN |
| **Backbone** | ResNet50 + Feature Pyramid Network (FPN) |
| **Pre-trained Dataset** | COCO (80 object classes) |
| **Framework** | PyTorch + TorchVision |
| **Device** | GPU (CUDA) if available, otherwise CPU |
| **Download Size** | ~170 MB |
| **Performance** | ~88-92% mAP on COCO validation set |

## Pipeline Stages

### Stage 1: Preprocessing

Input validation and cleaning:
- Scans input for supported image formats
- Validates each image (checks for corruption)
- Removes completely blank images
- Converts to RGB color space
- Temporary processed images are cleaned up automatically

### Stage 2: Instance Segmentation

Detection and segmentation:
- Loads pre-trained Mask R-CNN model
- Processes each cleaned image
- Detects object instances with confidence scores
- Generates pixel-level masks for each instance
- Creates colored overlay visualization

### Stage 3: Results Organization

Output generation:
- Saves colored segmentation images
- Generates JSON metadata files
- Creates summary results file

## Performance & Speed

| Factor | Impact |
|--------|--------|
| **Image Size** | Larger images = longer processing time |
| **Number of Objects** | More objects = longer per-image processing |
| **GPU vs CPU** | GPU: ~1-3 sec/image | CPU: ~10-30 sec/image |
| **Memory Usage** | ~2GB RAM minimum | More for very large images |

**Typical Processing Times (CPU):**
- Single small image (640×480): 5-10 seconds
- Single large image (1920×1080): 15-30 seconds
- Batch of 100 images: 15-30 minutes

## Troubleshooting

### "No supported images found" Error

**Problem**: Input folder has no valid images

**Solution**:
- Verify file extensions are supported (.jpg, .png, .bmp, .tiff, .webp)
- Check if images are corrupted (try opening in image viewer)
- Ensure you're pointing to the correct directory

### Model Download Fails

**Problem**: Cannot download pre-trained weights

**Solution**:
- Check your internet connection
- Weights are cached in `~/.cache/torch/hub/` after first download
- Manual download: https://github.com/pytorch/vision/releases

### Out of Memory Error

**Problem**: Pipeline runs out of RAM/VRAM

**Solution**:
- Reduce image resolution before processing
- Process images in smaller batches
- Use GPU if available (much more efficient)
- Close other applications

### Slow Processing (CPU)

**Problem**: Pipeline is very slow

**Solution**:
- Install GPU support if possible (CUDA for NVIDIA GPUs)
- Process smaller images
- Batch multiple images and process overnight

## Advanced Usage

### Running Individual Stages

**Preprocess only:**
```bash
python pipeline/preprocess.py --input ./images --output ./cleaned
```

**Segment only:**
```bash
python pipeline/segment.py --input ./images --images ./output/images --labels ./output/labels
```

### Adjusting Detection Threshold

Edit `pipeline/segment.py` and modify the `score_threshold` parameter (default: 0.5):
- Lower values (0.3-0.4): Detect more objects, including weak detections
- Higher values (0.6-0.8): Detect only confident instances

## Output Colors Reference

Instances are colored in this order:
1. Red
2. Blue
3. Cyan
4. Green
5. Yellow
6. Magenta
7. Orange
8. Purple
9. Light Blue
10. Pink

(Colors cycle if more than 10 instances)

## File Structure

```
Image-Segmentation/
├── pipeline/
│   ├── pipeline.py         # Main orchestrator
│   ├── preprocess.py       # Image preprocessing
│   └── segment.py          # Segmentation module
├── image_data/
│   └── sample.jpg          # Example image
├── requirements.txt        # Dependencies
├── README.md              # This file
└── pipeline_output/       # Generated outputs
    └── maskrcnn/
        ├── images/
        ├── labels/
        └── output.json
```

## FAQ

**Q: Can I use custom trained models?**
A: Currently uses pre-trained COCO weights. Modify `segment.py` to load custom weights.

**Q: What does the confidence score mean?**
A: It's the model's confidence (0-1) that the detected instance is correct. Higher = more confident.

**Q: How many instances can it detect?**
A: No hard limit, but more instances = slower processing. Typical images have 1-20 instances.

**Q: Can I process videos?**
A: Not currently. Extract frames first using ffmpeg, then process as images.

## License

Uses pre-trained models from PyTorch/TorchVision (BSD License)

## Support

For issues or questions, check:
1. Ensure all dependencies are installed
2. Verify your images are valid (not corrupted)
3. Check that paths are correct and accessible
4. Try with the sample image first


