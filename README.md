# Certificate Verification System

A rule-based certificate verification module that uses OCR (Optical Character Recognition) to extract text from certificate images and classify them using keyword matching. **No machine learning models are used** - this is a purely rule-based approach.

## Features

- 📸 **Image Upload**: Support for JPG, JPEG, and PNG formats
- 🔍 **OCR Text Extraction**: Uses Tesseract OCR to extract text from images
- 🎯 **Keyword-Based Classification**: Classifies certificates using predefined keyword matching
- ✅ **Verification Status**: Provides Accept/Not Verified status based on classification
- 🎨 **Modern Web UI**: Clean and user-friendly interface

## Supported Certificate Types

1. **Aadhar Card (UIDAI)**
   - Keywords: uidai, unique identification authority of india, aadhar, aadhaar, government of india

2. **Community Certificate**
   - Keywords: community certificate, scheduled caste, scheduled tribe, most backward class, backward class, revenue department, tahsildar

3. **Income Certificate**
   - Keywords: income certificate, annual income, revenue department, issued by, tahsildar, family income

4. **Educational Certificate (Anna University)**
   - Keywords: anna university, degree, b.e, b.tech, university, controller of examinations, chennai

5. **Unknown Certificate**
   - If no keywords match, the certificate is classified as "Unknown Certificate"

## Prerequisites

### 1. Python
- Python 3.8 or higher
- Download from: https://www.python.org/downloads/

### 2. Tesseract OCR
**For Windows:**
- Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
- Recommended version: Latest stable release
- During installation, make sure to check "Add to PATH" option
- Or note the installation path (usually `C:\Program Files\Tesseract-OCR\`)

**For Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**For macOS:**
```bash
brew install tesseract
```

### 3. Verify Tesseract Installation
Open Command Prompt (Windows) or Terminal (Linux/Mac) and run:
```bash
tesseract --version
```

If you see version information, Tesseract is installed correctly.

## Installation

### Step 1: Clone or Download the Project
Navigate to the project directory:
```bash
cd C:\Users\ADMIN\Desktop\CIP
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Step 4: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Configure Tesseract Path (Windows Only)
If Tesseract is not in your system PATH, edit `app.py` and uncomment/modify this line:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```
Replace the path with your actual Tesseract installation path.

## Running the Application

### Step 1: Start the Flask Server
```bash
python app.py
```

You should see output like:
```
============================================================
Certificate Verification System - Starting...
============================================================

Make sure Tesseract OCR is installed and in your system PATH.
For Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

Server will start at: http://127.0.0.1:5000
============================================================
 * Running on http://0.0.0.0:5000
```

### Step 2: Open Web Browser
Navigate to: **http://127.0.0.1:5000**

### Step 3: Upload Certificate Image
1. Click "Choose File" or drag and drop an image
2. Select a certificate image (JPG, JPEG, or PNG)
3. Click "Verify Certificate"
4. Wait for processing (OCR may take a few seconds)
5. View the results:
   - Certificate Type
   - Verification Status
   - Matched Keywords (if any)
   - Extracted Text

## Project Structure

```
CIP/
├── app.py                 # Flask backend with OCR and classification logic
├── templates/
│   └── index.html        # Frontend web interface
├── uploads/              # Temporary storage for uploaded images (auto-created)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## How It Works

### 1. Image Preprocessing
- Converts image to grayscale
- Applies adaptive thresholding for better text extraction
- Handles varying lighting conditions

### 2. OCR Text Extraction
- Uses Tesseract OCR engine
- Extracts all text from the preprocessed image
- Returns cleaned text string

### 3. Keyword Matching
- Converts extracted text to lowercase (case-insensitive)
- Checks for predefined keywords in each certificate category
- If multiple categories match, selects the one with highest priority
- If no keywords match, classifies as "Unknown Certificate"

### 4. Status Determination
- **Accepted**: Certificate type was successfully identified
- **Not Verified**: Certificate type is unknown or could not be verified

## Troubleshooting

### Issue: "TesseractNotFoundError"
**Solution:** 
- Make sure Tesseract is installed
- Add Tesseract to system PATH, or
- Set the path manually in `app.py` (see Installation Step 5)

### Issue: "No module named 'cv2'"
**Solution:**
```bash
pip install opencv-python
```

### Issue: "No module named 'pytesseract'"
**Solution:**
```bash
pip install pytesseract
```

### Issue: Poor OCR Accuracy
**Solutions:**
- Use high-quality, clear images
- Ensure text is not too small or blurry
- Try different image formats (PNG often works better than JPG)
- Ensure good lighting and contrast in the original image

### Issue: Port Already in Use
**Solution:**
- Change the port in `app.py`:
  ```python
  app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
  ```

## Customization

### Adding New Certificate Types
Edit `CERTIFICATE_KEYWORDS` dictionary in `app.py`:

```python
CERTIFICATE_KEYWORDS = {
    'Your Certificate Type': {
        'keywords': [
            'keyword1',
            'keyword2',
            'keyword3'
        ],
        'priority': 5  # Higher number = higher priority
    },
    # ... existing types
}
```

### Modifying OCR Settings
In `extract_text_ocr()` function, you can modify:
- PSM (Page Segmentation Mode): Currently set to 6
- OEM (OCR Engine Mode): Currently set to 3
- See Tesseract documentation for more options

## Technical Details

- **Backend**: Flask (Python web framework)
- **OCR Engine**: Tesseract OCR (via pytesseract)
- **Image Processing**: OpenCV
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Text Processing**: Python string operations

## Limitations

- Rule-based approach may not handle variations in certificate formats
- OCR accuracy depends on image quality
- Requires exact keyword matches (case-insensitive)
- Does not verify certificate authenticity, only classifies type

## License

This project is for educational/academic purposes.

## Author

Certificate Verification System - Rule-Based OCR Module

## Date

January 2026
