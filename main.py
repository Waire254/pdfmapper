import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import json
import uuid

def extract_tables_from_pdf(pdf_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    
    tables = []
    for i, image in enumerate(images):
        # Convert PIL Image to OpenCV format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Combine horizontal and vertical lines
        table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)

        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 100:  # Filter small contours
                table = {
                    "id": str(uuid.uuid4()),
                    "label_id": f"table_{i}_{len(tables)}",
                    "label": "table",
                    "ocr_text": "table",
                    "score": 1.0,
                    "xmin": x,
                    "xmax": x + w,
                    "ymin": y,
                    "ymax": y + h,
                    "type": "table",
                    "cells": extract_cells(img[y:y+h, x:x+w], x, y)
                }
                tables.append(table)

    return tables

def extract_cells(table_img, table_x, table_y):
    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine horizontal and vertical lines
    grid = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Find contours
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # Filter small contours
            cell_img = table_img[y:y+h, x:x+w]
            text = pytesseract.image_to_string(cell_img).strip()
            
            cell = {
                "id": str(uuid.uuid4()),
                "row": i // 10,  # Assuming 10 cells per row, adjust as needed
                "col": i % 10,
                "row_span": 1,
                "col_span": 1,
                "label": "",
                "xmin": table_x + x,
                "ymin": table_y + y,
                "xmax": table_x + x + w,
                "ymax": table_y + y + h,
                "score": 1.0,
                "text": text,
                "row_label": f"row_{i // 10}",
                "verification_status": "unverified",
                "status": "active",
                "failed_validation": "",
                "label_id": f"cell_{i}",
                "lookup_edited": False
            }
            cells.append(cell)

    return cells

def save_to_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

# Usage
pdf_path = 'pdf/file.pdf'
output_path = 'output.json'

extracted_tables = extract_tables_from_pdf(pdf_path)
save_to_json(extracted_tables, output_path)

print(f"Tables extracted and saved to {output_path}")
