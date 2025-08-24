import streamlit as st
import pdfplumber
from groq import Groq
import os, json
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as PlatypusImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import re

load_dotenv()
groqapi = os.getenv("GROQ_APIKEY")
groq_client = Groq(api_key=groqapi)

def get_report_type(report):
    report_type = report.get("report_type", "").lower()
    test_names = [test.get("test_name", "").lower() for test in report.get("test_results", [])]
    if any(keyword in report_type for keyword in ["blood", "hematology", "serum", "plasma", "cbc", "lipid", "glucose"]) or \
       any(test_name in ["hemoglobin", "wbc count", "rbc count", "platelet count", "glucose", "cholesterol"] 
           for test_name in test_names):
        return "blood"
    elif any(keyword in report_type for keyword in ["urine", "urinalysis", "ua"]) or \
         any(test_name in ["urine color", "urine ph", "specific gravity", "leukocytes", "nitrite", "protein", "glucose in urine", "ketones"] 
             for test_name in test_names):
        return "urine"
    elif any(keyword in report_type for keyword in ["x-ray", "xray", "mri", "ct scan", "ultrasound", "imaging", "radiograph", "sonogram"]) or \
         any("impression" in test_name or "finding" in test_name for test_name in test_names):
        return "imaging"
    elif any(keyword in report_type for keyword in ["pathology", "histology", "biopsy", "cytology"]) or \
         any("specimen" in test_name or "tissue" in test_name for test_name in test_names):
        return "pathology"
    else:
        return "other"

def is_blood_test(report):
    return get_report_type(report) == "blood"

def is_urine_test(report):
    return get_report_type(report) == "urine"

def is_imaging_report(report):
    return get_report_type(report) == "imaging"

def is_pathology_report(report):
    return get_report_type(report) == "pathology"

def parse_value_with_units(value_str):
    if not value_str:
        return None
    clean_str = value_str.replace(",", "")
    match = re.search(r'(\d+\.?\d*)', clean_str)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None

def parse_range(range_str):
    if not range_str:
        return None, None
    if range_str.startswith('<'):
        try:
            max_val = parse_value_with_units(range_str[1:].strip())
            return 0, max_val
        except:
            return None, None
    elif range_str.startswith('>'):
        try:
            min_val = parse_value_with_units(range_str[1:].strip())
            return min_val, min_val * 2
        except:
            return None, None
    range_str = range_str.replace("–", "-").replace("—", "-").replace(" - ", "-")
    if "-" in range_str:
        parts = range_str.split("-")
        if len(parts) >= 2:
            min_val = parse_value_with_units(parts[0].strip())
            max_val = parse_value_with_units(parts[1].strip())
            return min_val, max_val
    return None, None

def add_urine_test_visualization(report, elements):
    styles = getSampleStyleSheet()
    test_results = report.get("test_results", [])
    if not test_results:
        elements.append(Paragraph("<b>Note:</b> No urine test data found for visualization.", styles['Normal']))
        elements.append(Spacer(1, 10))
        return
    elements.append(Paragraph("<b>Urine Test Results:</b>", styles['Heading3']))
    elements.append(Spacer(1, 8))
    header = ["Parameter", "Result", "Reference Range", "Status"]
    rows = [header]
    color_mapping = {
        "color": {
            "normal": ["pale yellow", "yellow", "straw", "amber", "clear"],
            "abnormal": ["red", "brown", "orange", "green", "blue", "cloudy", "turbid"]
        },
        "clarity": {
            "normal": ["clear", "transparent"],
            "abnormal": ["cloudy", "turbid", "hazy"]
        },
        "ph": {
            "normal": lambda v: 4.5 <= parse_value_with_units(v) <= 8.0 if parse_value_with_units(v) is not None else False
        },
        "specific gravity": {
            "normal": lambda v: 1.005 <= parse_value_with_units(v) <= 1.030 if parse_value_with_units(v) is not None else False
        },
        "glucose": {
            "normal": ["negative", "none", "0", "normal"],
            "abnormal": ["positive", "trace", "1+", "2+", "3+", "4+"]
        },
        "protein": {
            "normal": ["negative", "none", "0", "normal"],
            "abnormal": ["positive", "trace", "1+", "2+", "3+", "4+"]
        },
        "ketones": {
            "normal": ["negative", "none", "0", "normal"],
            "abnormal": ["positive", "trace", "1+", "2+", "3+", "4+"]
        },
        "blood": {
            "normal": ["negative", "none", "0", "normal"],
            "abnormal": ["positive", "trace", "1+", "2+", "3+", "4+"]
        },
        "nitrite": {
            "normal": ["negative", "none", "0", "normal"],
            "abnormal": ["positive"]
        },
        "leukocytes": {
            "normal": ["negative", "none", "0", "normal"],
            "abnormal": ["positive", "trace", "1+", "2+", "3+", "4+"]
        },
        "bacteria": {
            "normal": ["negative", "none", "0", "normal", "not seen"],
            "abnormal": ["positive", "present", "few", "moderate", "many"]
        },
        "epithelial cells": {
            "normal": ["negative", "none", "0", "normal", "few", "occasional"],
            "abnormal": ["moderate", "many"]
        }
    }
    for test in report.get("test_results", []):
        test_name = (test.get("test_name") or "").strip().lower()
        value_raw = test.get("value") or ""
        value = str(value_raw).strip().lower() if not isinstance(value_raw, float) else str(value_raw).lower()
        ref_range = str(test.get("reference_range") or "").strip()
        if not test_name or not value:
            continue
        status = "Normal"
        for param, rules in color_mapping.items():
            if param in test_name:
                if "normal" in rules and callable(rules["normal"]):
                    if not rules["normal"](value):
                        status = "Abnormal"
                elif "normal" in rules and isinstance(rules["normal"], list):
                    if not any(normal_val in value for normal_val in rules["normal"]) or \
                       any(abnormal_val in value for abnormal_val in rules.get("abnormal", [])):
                        status = "Abnormal"
                break
        else:
            if ref_range and value:
                num_value = parse_value_with_units(value)
                min_val, max_val = parse_range(ref_range)
                if num_value is not None and min_val is not None and max_val is not None:
                    if num_value < min_val or num_value > max_val:
                        status = "Abnormal"
        rows.append([test_name.title(), value, ref_range, status])
    if len(rows) > 1:
        table = Table(rows, colWidths=[120, 100, 120, 80], hAlign="CENTER")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("ALIGN", (0, 1), (-1, -1), "CENTER"),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        for i in range(1, len(rows)):
            status = rows[i][3]
            if status == "Normal":
                table.setStyle(TableStyle([("BACKGROUND", (3, i), (3, i), colors.lightgreen)]))
            else:
                table.setStyle(TableStyle([("BACKGROUND", (3, i), (3, i), colors.lightcoral)]))
        elements.append(table)
        elements.append(Spacer(1, 20))
        for test in test_results:
            if "color" in (test.get("test_name") or "").lower():
                color_value = (test.get("value") or "").lower()
                if color_value:
                    elements.append(Paragraph("<b>Urine Color Representation:</b>", styles['Heading3']))
                    elements.append(Spacer(1, 8))
                    color_hex = {
                        "pale yellow": "#FFFFA0",
                        "yellow": "#FFFF00",
                        "dark yellow": "#CCCC00",
                        "amber": "#FFBF00",
                        "orange": "#FFA500",
                        "red": "#FF0000",
                        "pink": "#FFC0CB",
                        "brown": "#A52A2A",
                        "clear": "#F0F8FF",
                        "cloudy": "#E6E6FA"
                    }
                    color_code = "#FFFFA0"
                    for color_name, hex_code in color_hex.items():
                        if color_name in color_value:
                            color_code = hex_code
                            break
                    color_table = Table([[""], [""], [""]], colWidths=[100], rowHeights=[50, 50, 50])
                    color_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (0, 2), colors.HexColor(color_code)),
                        ("BOX", (0, 0), (0, 2), 1, colors.black),
                    ]))
                    elements.append(color_table)
                    elements.append(Spacer(1, 10))
                    elements.append(Paragraph(f"<i>Reported color: {color_value}</i>", styles['Normal']))
                    elements.append(Spacer(1, 20))
                    break

def add_imaging_report_visualization(report, elements):
    styles = getSampleStyleSheet()
    findings = None
    impression = None
    for test in report.get("test_results", []):
        test_name = (test.get("test_name") or "").lower()
        if "finding" in test_name:
            findings = test.get("value")
        elif "impression" in test_name:
            impression = test.get("value")
    if not findings and not impression and "doctor_notes" in report:
        notes = report.get("doctor_notes", "")
        if "finding" in notes.lower():
            findings_section = re.search(r'(?i)findings?:(.+?)(?:impression:|assessment:|conclusion:|$)', notes, re.DOTALL)
            if findings_section:
                findings = findings_section.group(1).strip()
        if "impression" in notes.lower():
            impression_section = re.search(r'(?i)impression:(.+?)(?:recommendation:|plan:|$)', notes, re.DOTALL)
            if impression_section:
                impression = impression_section.group(1).strip()
    if findings:
        elements.append(Paragraph("<b>Findings:</b>", styles['Heading3']))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(findings, styles['Normal']))
        elements.append(Spacer(1, 15))
    if impression:
        elements.append(Paragraph("<b>Impression:</b>", styles['Heading3']))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(impression, styles['Normal']))
        elements.append(Spacer(1, 15))
    if not findings and not impression:
        elements.append(Paragraph("<b>Note:</b> No structured findings or impressions found in this imaging report.", 
                                styles['Normal']))
        elements.append(Spacer(1, 10))

def add_pathology_report_visualization(report, elements):
    styles = getSampleStyleSheet()
    specimen = None
    diagnosis = None
    microscopic = None
    for test in report.get("test_results", []):
        test_name = (test.get("test_name") or "").lower()
        if "specimen" in test_name:
            specimen = test.get("value")
        elif "diagnosis" in test_name:
            diagnosis = test.get("value")
        elif "microscopic" in test_name:
            microscopic = test.get("value")
    if not any([specimen, diagnosis, microscopic]) and "doctor_notes" in report:
        notes = report.get("doctor_notes", "")
        if "specimen" in notes.lower():
            specimen_section = re.search(r'(?i)specimen:(.+?)(?:clinical|gross|microscopic|diagnosis:|$)', notes, re.DOTALL)
            if specimen_section:
                specimen = specimen_section.group(1).strip()
        if "diagnosis" in notes.lower():
            diagnosis_section = re.search(r'(?i)diagnosis:(.+?)(?:comment:|note:|$)', notes, re.DOTALL)
            if diagnosis_section:
                diagnosis = diagnosis_section.group(1).strip()
        if "microscopic" in notes.lower():
            microscopic_section = re.search(r'(?i)microscopic:(.+?)(?:diagnosis:|assessment:|$)', notes, re.DOTALL)
            if microscopic_section:
                microscopic = microscopic_section.group(1).strip()
    if specimen:
        elements.append(Paragraph("<b>Specimen:</b>", styles['Heading3']))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(specimen, styles['Normal']))
        elements.append(Spacer(1, 15))
    if diagnosis:
        elements.append(Paragraph("<b>Diagnosis:</b>", styles['Heading3']))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(diagnosis, styles['Normal']))
        elements.append(Spacer(1, 15))
    if microscopic:
        elements.append(Paragraph("<b>Microscopic Description:</b>", styles['Heading3']))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(microscopic, styles['Normal']))
        elements.append(Spacer(1, 15))
    if not any([specimen, diagnosis, microscopic]):
        elements.append(Paragraph("<b>Note:</b> No structured pathology data found in this report.", 
                                styles['Normal']))
        elements.append(Spacer(1, 10))

def add_generic_report_visualization(report, elements):
    styles = getSampleStyleSheet()
    elements.append(Paragraph("<b>Note:</b> This is a general medical report without specific visualization.", 
                            styles['Normal']))
    elements.append(Spacer(1, 15))
    test_names = []
    values = []
    units = []
    ref_ranges = []
    for test in report.get("test_results", []):
        test_name = (test.get("test_name") or "").strip()
        value_raw = test.get("value") or ""
        value_str = str(value_raw).strip() if not isinstance(value_raw, float) else str(value_raw)
        unit = str(test.get("unit") or "").strip()
        ref_range = str(test.get("reference_range") or "").strip()
        if test_name and value_str:
            test_names.append(test_name)
            values.append(value_str)
            units.append(unit)
            ref_ranges.append(ref_range)
    if test_names:
        elements.append(Paragraph("<b>Test Results Summary:</b>", styles['Heading3']))
        elements.append(Spacer(1, 8))
        header = ["Test", "Result", "Unit", "Reference Range"]
        rows = [header]
        for i in range(len(test_names)):
            rows.append([test_names[i], values[i], units[i], ref_ranges[i]])
        table = Table(rows, colWidths=[120, 100, 80, 120], hAlign="CENTER")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#9b59b6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("ALIGN", (0, 1), (-1, -1), "CENTER"),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

def add_blood_test_bargraph(report, elements):
    test_names = []
    actual_values = []
    normal_mins = []
    normal_maxs = []
    original_values = []
    original_ranges = []
    for test in report.get("test_results", []):
        test_name = test.get("test_name", "").strip()
        value_str = str(test.get("value", "")) if not isinstance(test.get("value"), float) else str(test.get("value", ""))
        ref_str = str(test.get("reference_range", ""))
        actual_val = parse_value_with_units(value_str)
        if actual_val is None:
            continue
        ref_min, ref_max = parse_range(ref_str)
        if ref_min is None or ref_max is None:
            continue
        test_names.append(test_name)
        actual_values.append(actual_val)
        normal_mins.append(ref_min)
        normal_maxs.append(ref_max)
        original_values.append(value_str)
        original_ranges.append(ref_str)
    if not test_names:
        elements.append(Paragraph("<b>Note:</b> No valid numerical data found for blood test chart generation.", 
                                getSampleStyleSheet()['Normal']))
        elements.append(Spacer(1, 10))
        return
    max_tests_per_chart = 4
    num_charts = (len(test_names) + max_tests_per_chart - 1) // max_tests_per_chart
    for chart_index in range(num_charts):
        start_idx = chart_index * max_tests_per_chart
        end_idx = min(start_idx + max_tests_per_chart, len(test_names))
        chart_test_names = test_names[start_idx:end_idx]
        chart_actual_values = actual_values[start_idx:end_idx]
        chart_normal_mins = normal_mins[start_idx:end_idx]
        chart_normal_maxs = normal_maxs[start_idx:end_idx]
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(chart_test_names))
        width = 0.35
        normal_midpoints = [(min_val + max_val) / 2 for min_val, max_val in zip(chart_normal_mins, chart_normal_maxs)]
        bars1 = ax.bar(x - width/2, chart_actual_values, width, label='Your Values', 
                       color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, normal_midpoints, width, label='Normal Range (Average)', 
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        range_errors = [[(mid - min_val), (max_val - mid)] for mid, min_val, max_val in 
                       zip(normal_midpoints, chart_normal_mins, chart_normal_maxs)]
        range_errors = np.array(range_errors).T
        ax.errorbar(x + width/2, normal_midpoints, yerr=range_errors, 
                    fmt='none', color='black', capsize=5, alpha=0.7)
        chart_title = 'Your Blood Test Results vs Normal Range'
        if num_charts > 1:
            chart_title += f' (Chart {chart_index + 1} of {num_charts})'
        ax.set_xlabel('Blood Tests', fontsize=12, fontweight='bold')
        ax.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax.set_title(chart_title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(chart_test_names, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for i, (bar1, bar2, actual, normal_mid, min_val, max_val) in enumerate(zip(
                bars1, bars2, chart_actual_values, normal_midpoints, chart_normal_mins, chart_normal_maxs)):
            ax.text(bar1.get_x() + bar1.get_width()/2., actual + max(chart_actual_values) * 0.01,
                    f'{actual:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(bar2.get_x() + bar2.get_width()/2., normal_mid + max(normal_midpoints) * 0.01,
                    f'{min_val:.1f}-{max_val:.1f}', ha='center', va='bottom', 
                    fontsize=8, rotation=0)
            if actual < min_val or actual > max_val:
                bars1[i].set_color('#e74c3c')
            else:
                bars1[i].set_color('#2ecc71')
        plt.tight_layout()
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format="png", dpi=150, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        elements.append(Spacer(1, 15))
        chart_heading = "<b>Blood Test Comparison Chart</b>"
        if num_charts > 1:
            chart_heading += f" <i>(Chart {chart_index + 1} of {num_charts})</i>"
        elements.append(Paragraph(chart_heading, getSampleStyleSheet()['Heading3']))
        elements.append(Spacer(1, 8))
        elements.append(PlatypusImage(img_buffer, width=550, height=330))
        elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>Values Comparison Table:</b>", getSampleStyleSheet()['Heading3']))
    elements.append(Spacer(1, 8))
    comparison_header = ["Test Name", "Your Value", "Normal Range", "Status"]
    comparison_rows = [comparison_header]
    for i, (test_name, original_val, original_range, actual_val, min_val, max_val) in enumerate(
        zip(test_names, original_values, original_ranges, actual_values, normal_mins, normal_maxs)):
        if actual_val < min_val:
            status = "Below Normal"
        elif actual_val > max_val:
            status = "Above Normal"
        else:
            status = "Normal"
        comparison_rows.append([test_name, original_val, original_range, status])
    comparison_table = Table(comparison_rows, colWidths=[120, 80, 100, 80], hAlign="CENTER")
    comparison_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ALIGN", (0, 1), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
    ]))
    for i in range(1, len(comparison_rows)):
        status = comparison_rows[i][3]
        if status == "Normal":
            comparison_table.setStyle(TableStyle([("BACKGROUND", (3, i), (3, i), colors.lightgreen)]))
        elif status == "Below Normal":
            comparison_table.setStyle(TableStyle([("BACKGROUND", (3, i), (3, i), colors.lightyellow)]))
        else:
            comparison_table.setStyle(TableStyle([("BACKGROUND", (3, i), (3, i), colors.lightcoral)]))
    elements.append(comparison_table)
    elements.append(Spacer(1, 20))

def generate_pdf(parsed_reports, output_file):
    styles = getSampleStyleSheet()
    elements = []
    doc = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=60,
        bottomMargin=60
    )
    for i, report in enumerate(parsed_reports):
        elements.append(Paragraph("<b>Medical Report</b>", styles['Title']))
        elements.append(Spacer(1, 18))
        if "patient_info" in report:
            patient_info = report["patient_info"]
            pat_text = "<br/>".join([
                f"<b>Name:</b> {patient_info.get('name', 'N/A')}",
                f"<b>Age:</b> {patient_info.get('age', 'N/A')}",
                f"<b>Sex:</b> {patient_info.get('sex', 'N/A')}"
            ])
            elements.append(Paragraph(pat_text, styles['Normal']))
            elements.append(Spacer(1, 18))
        elements.append(Paragraph(
            f"<b>Report Type:</b> {report.get('report_type','Unknown')}",
            styles['Heading2']
        ))
        elements.append(Spacer(1, 18))
        if "test_results" in report and report["test_results"]:
            header = ["Test Name", "Value", "Unit", "Reference Range"]
            all_rows = [header]
            for test in report["test_results"]:
                test_name = (test.get("test_name") or "").strip()
                value_raw = test.get("value") or ""
                value = str(value_raw).strip() if not isinstance(value_raw, float) else str(value_raw)
                unit = str(test.get("unit") or "").strip()
                ref_range = str(test.get("reference_range") or "").strip()
                if not test_name or (not value and not unit and not ref_range):
                    continue
                all_rows.append([test_name, value, unit, ref_range])
            if len(all_rows) > 1:
                max_rows = 6
                chunks = [all_rows[i:i + max_rows] for i in range(0, len(all_rows), 5)]
                for idx, chunk in enumerate(chunks):
                    table = Table(chunk, colWidths=[120, 100, 80, 120], hAlign="CENTER")
                    header_color = colors.HexColor("#4CAF50") if idx % 2 == 0 else colors.HexColor("#2196F3")
                    table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), header_color),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 11),
                        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 1), (-1, -1), 10),
                        ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ]))
                    elements.append(table)
                    elements.append(Spacer(1, 20))
        report_type = get_report_type(report)
        if report_type == "blood":
            add_blood_test_bargraph(report, elements)
        elif report_type == "urine":
            add_urine_test_visualization(report, elements)
        elif report_type == "imaging":
            add_imaging_report_visualization(report, elements)
        elif report_type == "pathology":
            add_pathology_report_visualization(report, elements)
        else:
            add_generic_report_visualization(report, elements)
        if "doctor_notes" in report:
            elements.append(Paragraph("<b>Doctor Notes:</b>", styles['Heading3']))
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(report["doctor_notes"], styles['Normal']))
            elements.append(Spacer(1, 25))
        if i < len(parsed_reports) - 1:
            elements.append(PageBreak())
    doc.build(elements)
    return output_file

st.title("Medical Report Parser (Multi-PDF)")

files = st.file_uploader("Upload your reports", type=["pdf"], accept_multiple_files=True)

def extract_test_results(reports, format_type="dict"):
    results = {}
    if not isinstance(reports, list):
        reports = [reports]
    for report in reports:
        if not report or "test_results" not in report:
            continue
        for test in report.get("test_results", []):
            test_name = test.get("test_name", "").strip()
            if not test_name:
                continue
            value_raw = test.get("value", "")
            value = str(value_raw).strip() if not isinstance(value_raw, float) else str(value_raw)
            unit = test.get("unit", "")
            if unit:
                value = f"{value} {unit}"
            results[test_name] = value
    if format_type.lower() == "json":
        import json
        return json.dumps(results, indent=2)
    return results

def merge_reports(reports):
    if not reports:
        return []
    if len(reports) == 1:
        return reports
    merged_report = {
        "patient_info": {},
        "report_type": "combined",
        "test_results": [],
        "doctor_notes": ""
    }
    unique_tests = set()
    doctor_notes = []
    for report in reports:
        if report.get("patient_info") and any(report["patient_info"].values()):
            merged_report["patient_info"] = report["patient_info"]
            break
    for report in reports:
        if report.get("doctor_notes"):
            report_type = report.get("report_type", "Unknown")
            doctor_notes.append(f"[{report_type.upper()} REPORT] {report['doctor_notes']}")
        for test in report.get("test_results", []):
            test_name = str(test.get("test_name", "")).strip().lower()
            if not test_name or test_name in unique_tests:
                continue
            unique_tests.add(test_name)
            merged_report["test_results"].append(test)
    merged_report["doctor_notes"] = "\n\n".join(doctor_notes)
    return [merged_report]

if files:
    all_reports = []
    for fileupload in files:
        with pdfplumber.open(fileupload) as pdf:
            pdf_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        max_text_length = 2500
        truncated_pdf_text = pdf_text[:max_text_length] if len(pdf_text) > max_text_length else pdf_text
        if len(truncated_pdf_text) < len(pdf_text):
            st.warning(f"The PDF text was truncated from {len(pdf_text)} to {len(truncated_pdf_text)} characters to avoid token limit errors.")
        system_prompt = """Extract structured medical data as JSON with schema:
{
  "patient_info": {"name": string, "age": number, "sex": string},
  "report_type": string,
  "test_results": [{"test_name": string, "value": string, "unit": string, "reference_range": string}],
  "doctor_notes": string
  "summary" : string
}

For blood/urine tests: Include parameters, values, units, ranges.
For imaging/pathology: Include findings, impressions, specimen details, diagnosis."""
        try:
            completion = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this medical report into structured JSON:\n\n{truncated_pdf_text}"}
                ],
                response_format={"type": "json_object"}
            )
            parsed_data = json.loads(completion.choices[0].message.content)
            if isinstance(parsed_data, dict):
                parsed_data = [parsed_data]
        except groq.APIStatusError as e:
            st.error(f"API Error: {str(e)}")
            continue
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            continue
        all_reports.extend(parsed_data)
    test_results_only = extract_test_results(all_reports)
    test_results_json = extract_test_results(all_reports, format_type="json")
    st.subheader("Test Results JSON")
    if test_results_only:
        st.json(test_results_only)
        if st.button("Download Test Results as JSON"):
            st.download_button(
                label="Download JSON",
                data=test_results_json,
                file_name="test_results.json",
                mime="application/json"
            )
    else:
        st.info("No test results found in the reports.")
    if len(files) > 1:
        merged_reports = merge_reports(all_reports)
        st.subheader("Merged Report (Duplicates Removed)")
        st.json(merged_reports)
        pdf_buffer = BytesIO()
        generate_pdf(merged_reports, pdf_buffer)
        pdf_buffer.seek(0)
        st.success(f"Successfully merged {len(all_reports)} reports into a single report with {len(merged_reports[0].get('test_results', []))} unique test results.")
    else:
        st.subheader("Report")
        st.json(all_reports)
        pdf_buffer = BytesIO()
        generate_pdf(all_reports, pdf_buffer)
        pdf_buffer.seek(0)
    st.download_button("📥 Download Final Report", pdf_buffer, file_name="final_report.pdf", mime="application/pdf")
