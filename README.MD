<div align="center">

  <img src="assets/affineart_logo.png" alt="Affine Logo" width="200"/>
  <h1>Affine Art - Image Editor</h1>

  <p>
    <b>Image processing engine built with Python, NumPy, Streamlit and OpenCV</b>
  </p>

  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Streamlit-UI-red?style=flat-square" />
  <img src="https://img.shields.io/badge/NumPy-1.26%2B-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/OpenCV-4.10%2B-blue?style=flat-square" />

</div>

## 📑 Table of Contents
1. [Overview](#-overview)
2. [Technical Concept](#technical-concept)
3. [Implementation Details](#implementation-details)
4. [Architecture](#-architecture)
5. [Installation](#-installation)
6. [User Guide](#-user-guide)
7. [Testing and Quality](#-testing--quality)

## 🔭 Overview
The main goal of AffineArt is to provide a flexible image-editing environment  
By separating the compute core from the UI layer, the application remains stable  
and can apply complex mathematical filters in real time  


## 🧠 Technical Concept

AffineArt is built around a simple idea:  
**every image edit should be represented as an operation in a processing pipeline**

Instead of modifying pixels directly, each action is stored as a structured operation with parameters  
The final image is always produced by **replaying the pipeline** from original image

### Why this approach?

* **No cumulative loss of quality** - operations are reapplied on the original image  
* **Easy undo/redo** - simply remove or reorder operations  
* **Deterministic results** - the same pipeline always produces the same output  
* **Extensibility** - new filters can be added without changing the UI logic  

### Core Components

- **Processing Engine (`apply.py`)**  
  Executes operations sequentially using NumPy and OpenCV

- **Operation Catalog (`catalog.py`)**  
  Central registry of all available operations

- **Validation Layer (`validation.py`)**  
  Ensures only valid parameter values reach the engine

- **Streamlit UI (`app.py`)**  
  Builds controls dynamically based on the catalog -  
  the interface adapts when new operations are added



## 🛠️ Implementation Details

### Pipeline-Based Processing

Each edit is represented as an `Operation` object and appended to pipeline  
The engine never mutates the original image - instead, it rebuilds result when needed

```python
def apply_pipeline(image, operations):
    result = image
    for op in operations:
        result = op.apply(result, op.params)
    return result
```

### Registry Pattern for Operations

Operations are registered once and discovered dynamically:

```python
operations = {}

def register_operation(op):
    operations[op.id] = op
```

`catalog.py` simply calls `register_operation(...)`, which makes adding new tools trivial:

> implement -> register -> appears in UI automatically


### Vectorized Math (NumPy + OpenCV)

Expensive operations rely on vectorization instead of Python loops:

```python
lut = (np.arange(256) / 255.0) ** gamma * 255
output = cv2.LUT(image, lut.astype("uint8"))
```

This provides real-time feedback even for high-resolution images

### Strong Parameter Validation

Before execution, each operation validates its inputs

### UI Generated From Metadata

The UI does not "know" about operations directly.  
Instead, it inspects each registered operation and builds widgets dynamically

### 🎨 Available Operations
| Category | Included Filters |
|----------|------------------|
| **Adjust** | Gamma Correction, Brightness, Contrast |
| **Geometry** | Rotation, Scaling, Translation |
| **Filters** | Gaussian Blur, Unsharp Mask (Sharpening) |
| **Colors** | Grayscale, Sepia, HSV Color Space Tuning |
| **Edges** | Canny Edge Detection |
| **Perspective** | Vertical Correction |


## 🛠️ Architecture

```text
AffineArt/
├── app.py                  # Presentation layer (Streamlit UI)
├── requirements.txt        # Project dependencies
├── core/                   # Business logic layer
│   ├── operations/         # Image processing algorithms
│   │   ├── adjust.py       # Gamma, brightness, contrast
│   │   ├── affine.py       # Rotations and scaling
│   │   ├── convert.py      # Grayscale and color conversions
│   │   ├── filters.py      # Blur and sharpening
│   │   ├── hsv.py          # Color space adjustments
│   │   ├── perspective.py  # Perspective transformations
│   │   └── edges.py        # Edge detection
│   ├── apply.py            # Processing Engine
│   ├── catalog.py          # Centralized operation metadata
│   ├── registry.py         # Registry
│   ├── schemes.py          # Data structures
│   └── validation.py       # Parameter validation logic
├── tests/                  # Automated tests
│   ├── test_apply.py       # Pipeline and logic tests
│   ├── test_codestyle.py   # Pylint and PEP8
│   └── test_operations.py  # Algorithm correctness tests
└── assets/                 # Project branding and logos
    └── affineart_logo.png  # AffineArt logo
```



## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/novykdan/AffineArt.git
cd AffineArt
```

### Create a virtual environment

```bash
py -3.12 -m venv venv
```

Activate it:

**Windows**
```bash
.\venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

🎉 After launching the app opens in your browser and is ready to use



## 📖 User Guide

AffineArt is designed to be simple: upload -> edit -> apply -> save

### Upload an image
Use the sidebar **Upload** widget to choose any PNG or JPG file  
Uploading a new image resets the current editing pipeline

### Choose an operation
Select:
- **category** (Adjust, Geometry, Filters, Colors, etc.)
- specific **operation**

Sidebar automatically shows available controls for that tool

### Adjust parameters
Use sliders and inputs to tune the effect 
Click **Preview** to see the result before adding it permanently  

### Apply the step
Click **Add** to add the operation to the processing pipeline  

### Undo when needed
Made a mistake? Click **Undo** - the last step is removed, and the result is rebuilt
Or use removal button on pipeline list

### Save the result
Use the download button to export the edited image (PNG or JPEG)


## ✅ Testing & Quality

```bash
python -m pytest
pytest --cov=core --cov-report=term-missing
```

> Core keeps >90% coverage; UI is tested manually


---

**Author:** Daniil Novykov  
*This project was developed as part of the BI-PYT course at FIT CTU (2025/2026).*