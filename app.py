"""
Certificate Verification Module - Rule-Based OCR System
========================================================
This module performs certificate verification using OCR and keyword matching.
No machine learning models are used - purely rule-based approach.

Author: Certificate Verification System
Date: 2026
"""

import os
import re
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import cv2
import pytesseract
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure Tesseract path (Windows - adjust if needed)
# For Windows, you may need to set the path manually:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Uncomment and adjust the path above if Tesseract is not in your system PATH

# ============================================================================
# CERTIFICATE KEYWORD DICTIONARY
# ============================================================================
# Priority order: Higher index = higher priority (checked first)
CERTIFICATE_KEYWORDS = {
    'Aadhar Card': {
        'keywords': [
            'uidai',
            'unique identification authority of india',
            'aadhar',
            'aadhaar',
            'government of india'
        ],
        'priority': 4
    },
    'Community Certificate': {
        'keywords': [
            'community certificate',
            'scheduled caste',
            'scheduled tribe',
            'most backward class',
            'backward class',
            'revenue department',
            'tahsildar'
        ],
        'priority': 3
    },
    'Income Certificate': {
        'keywords': [
            'income certificate',
            'annual income',
            'revenue department',
            'issued by',
            'tahsildar',
            'family income'
        ],
        'priority': 2
    },
    'Educational Certificate (Anna University)': {
        'keywords': [
            'anna university',
            'degree',
            'b.e',
            'b.tech',
            'university',
            'controller of examinations',
            'chennai'
        ],
        'priority': 1
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image_path):
    """
    Preprocess the certificate image for better OCR accuracy.
    
    Steps:
    1. Read image
    2. Convert to grayscale
    3. Apply thresholding (binary)
    4. Optional: Noise removal
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better text extraction
    # This helps with varying lighting conditions
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Alternative: Simple thresholding (uncomment if adaptive doesn't work well)
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh


def extract_text_ocr(image_path):
    """
    Extract text from certificate image using Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text from the image
    """
    try:
        # Preprocess image
        processed_img = preprocess_image(image_path)
        
        # Perform OCR using pytesseract
        # Using PSM mode 6: Assume uniform block of text
        # Using PSM mode 3: Fully automatic page segmentation (default)
        custom_config = r'--oem 3 --psm 6'
        extracted_text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        # Clean up extracted text
        extracted_text = extracted_text.strip()
        
        return extracted_text
    
    except Exception as e:
        raise Exception(f"OCR extraction failed: {str(e)}")


def classify_certificate(text):
    """
    Classify certificate type based on keyword matching.
    
    Logic:
    1. Convert text to lowercase for case-insensitive matching
    2. Check each certificate category for keyword matches
    3. If multiple matches, choose highest priority
    4. If no matches, return "Unknown Certificate"
    
    Args:
        text (str): Extracted OCR text
        
    Returns:
        tuple: (certificate_type, matched_keywords)
    """
    if not text or len(text.strip()) == 0:
        return "Unknown Certificate", []
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Track matches with their priorities
    matches = []
    
    # Check each certificate category
    for cert_type, cert_data in CERTIFICATE_KEYWORDS.items():
        keywords = cert_data['keywords']
        priority = cert_data['priority']
        
        # Check if any keyword matches
        matched_keywords = []
        for keyword in keywords:
            if keyword in text_lower:
                matched_keywords.append(keyword)
        
        # If any keywords matched, add to matches list
        if matched_keywords:
            matches.append({
                'type': cert_type,
                'keywords': matched_keywords,
                'priority': priority
            })
    
    # If no matches found
    if not matches:
        return "Unknown Certificate", []
    
    # Sort by priority (highest first) and return the top match
    matches.sort(key=lambda x: x['priority'], reverse=True)
    top_match = matches[0]
    
    return top_match['type'], top_match['keywords']


def determine_status(certificate_type):
    """
    Determine acceptance status based on certificate type.
    
    Args:
        certificate_type (str): Classified certificate type
        
    Returns:
        str: "Accepted" or "Not Verified"
    """
    if certificate_type == "Unknown Certificate":
        return "Not Verified"
    else:
        return "Accepted"


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and process certificate verification.
    
    Returns:
        JSON response with:
        - certificate_type: Type of certificate detected
        - status: Accepted / Not Verified
        - extracted_text: Full OCR text
        - matched_keywords: List of matched keywords (if any)
    """
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'certificate_type': 'Unknown Certificate',
                'status': 'Not Verified',
                'extracted_text': '',
                'matched_keywords': []
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'certificate_type': 'Unknown Certificate',
                'status': 'Not Verified',
                'extracted_text': '',
                'matched_keywords': []
            }), 400
        
        # Check if file extension is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Please upload JPG, JPEG, or PNG files.',
                'certificate_type': 'Unknown Certificate',
                'status': 'Not Verified',
                'extracted_text': '',
                'matched_keywords': []
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text using OCR
        extracted_text = extract_text_ocr(filepath)
        
        # Classify certificate
        certificate_type, matched_keywords = classify_certificate(extracted_text)
        
        # Determine status
        status = determine_status(certificate_type)
        
        # Clean up uploaded file (optional - remove if you want to keep files)
        # os.remove(filepath)
        
        # Return results
        return jsonify({
            'success': True,
            'certificate_type': certificate_type,
            'status': status,
            'extracted_text': extracted_text,
            'matched_keywords': matched_keywords
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'certificate_type': 'Unknown Certificate',
            'status': 'Not Verified',
            'extracted_text': '',
            'matched_keywords': []
        }), 500


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Certificate Verification System - Starting...")
    print("=" * 60)
    print("\nMake sure Tesseract OCR is installed and in your system PATH.")
    print("For Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    print("\nServer will start at: http://127.0.0.1:5000")
    print("=" * 60)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
