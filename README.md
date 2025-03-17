# Signature Detection with YOLO & Gradio

<table>
  <tr>
    <td>
      <a href="https://ultralytics.com/yolov8"><img src="https://img.shields.io/badge/YOLOv8-Ultralytics-76B900?style=for-the-badge&labelColor=black&logo=nvidia" alt="YOLOv8 Badge" /></a>
    </td>
    <td>
      <a href="https://gradio.app/"><img src="https://img.shields.io/badge/Gradio-3.x-2496ED?style=for-the-badge&labelColor=black&logo=python" alt="Gradio Badge" /></a>
    </td>
    <td>
      <a href="https://www.python.org/"><img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&labelColor=black&logo=python&logoColor=3776AB" alt="Python Badge" /></a>
    </td>
    <td>
      <a href="https://opencv.org/"><img src="https://img.shields.io/badge/-Opencv-5C3EE8?style=for-the-badge&labelColor=black&logo=opencv&logoColor=5C3EE8" alt="Opencv Badge" /></a>
    </td>
    <td>
      <a href="https://github.com/youruser/signature-detection"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT Badge" /></a>
    </td>
  </tr>
</table>

A complete pipeline for **detecting handwritten signatures** in documents (PDF or images) using **YOLO (Ultralytics)** and providing a **Gradio** interface for easy testing. Additionally, includes scripts for extracting certificate information from PDF files that may contain electronic signatures (X.509-based).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Running Gradio App](#1-running-gradio-app)
  - [2. Command-Line Scripts](#2-command-line-scripts)
- [Signature Extraction & Certificate Info](#signature-extraction--certificate-info)
- [Examples](#examples)
- [License](#license)
- [Contributing](#contributing)

---

## Overview

This repository aims to:
1. **Detect** any handwritten signatures in a scanned or digital document using a YOLO-based model (e.g., YOLOv8).
2. Provide a **Gradio UI** to upload or drop images/PDF pages and see detection results.
3. **Extract** certificate fields if a PDF is electronically signed (using `pyhanko` or `cryptography` to parse X.509 certificates).
4. Optionally **generate** a new PDF summarizing both the certificate information and the bounding boxes of the found signatures.

---

## Features

- **YOLOv8-based** detection for handwritten signatures with `onnx` or `.pt`.
- **PDF** to image conversion (via `pdf2image`) + optional text extraction.
- **Metadata** extraction: X.509 details, ID of doc, signers’ roles, etc. (with regex scanning).
- **Image Cropping**: Optionally store the region-of-interest (ROI) of the signature.
- **Gradio** interface for interactive testing.
- **Export** functionality to generate a new PDF with a summary/certificate of the detection.

---

## Project Structure

signature-detection/ ├── README.md ├── requirements.txt ├── app_gradio.py # Main Gradio app ├── Models/ │ └── yolov8sFT.onnx # Example YOLO model in ONNX ├── detection_scripts/ │ ├── detect_signature_in_images.py │ └── ... ├── electronicSignature/ │ ├── extract_certificates_from_pdf.py │ ├── extract_certificate_info.py │ ├── generate_certificate_pdf.py │ └── ... ├── utils/ │ ├── ... │ └── ... └── ...

yaml
Copier
Modifier

- **`app_gradio.py`**: Contains the Gradio Blocks for uploading an image, setting thresholds, etc.
- **`detection_scripts/detect_signature_in_images.py`**: The function that loads YOLO model, does detection, returns bounding boxes.
- **`electronicSignature/`**: Routines for extracting PDF certificates (X.509, pkcs7) and generating a summary PDF.
- **`utils/`**: Additional scripts (file management, logs, etc.)

---
## License
This project is licensed under the Apache Software License 2.0.

