# Medical Report Parser (Multi-PDF)

This project is a **Streamlit web app** that parses medical reports from PDFs, extracts structured data using Groq's LLM API, and generates summarized results in both JSON and PDF formats.

## Features

* **Multi-PDF Upload**: Upload one or multiple medical reports in PDF format.
* **LLM-Powered Parsing**: Uses Groq's Llama3 model to extract structured data from raw text.
* **Report Type Detection**: Automatically detects whether the report is:

  * Blood Test
  * Urine Test
  * Imaging Report (X-ray, MRI, CT, Ultrasound, etc.)
  * Pathology Report (biopsy, histology, cytology)
  * Other
* **Data Extraction**:

  * Extracts `patient_info`, `test_results`, `doctor_notes`, and `summary`.
  * Test results include values, units, and reference ranges.
* **Duplicate Removal**: When multiple reports are uploaded, duplicate test entries are removed in the merged report.
* **Output Options**:
  
  * Download a **final PDF report** with all structured data, visualizations, and summaries.

## Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **LLM Parsing**: [Groq API](https://groq.com/) with Llama3 model
* **PDF Handling**: [pdfplumber](https://github.com/jsvine/pdfplumber)
* **Report Generation**: [ReportLab](https://www.reportlab.com/) for structured PDF output
* **Visualization**: Matplotlib (for charts in reports)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/medical-report-parser.git
   cd medical-report-parser
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   * Create a `.env` file in the project root.
   * Add your Groq API key:

     ```env
     GROQ_APIKEY=your_api_key_here
     ```

## Usage

Run the Streamlit app:

```bash
streamlit run test7.py
```

1. Upload one or more PDF medical reports.
2. The app will parse them, extract structured JSON, and display test results.
3. Download:

   
   * **Final Structured Report (PDF)**

## Example Workflow

1. Upload `blood_report.pdf` and `urine_report.pdf`.
2. The app extracts:

   * Patient Info: Name, Age, Sex
   * Test Results: Hemoglobin, WBC count, Urine pH, etc.
   * Doctor Notes & Summary
3. JSON and PDF versions are available for download.

## Notes

* Large PDF text (>2500 characters) is truncated to avoid API token limits.
* The merged report combines multiple inputs while removing duplicate tests.
* The output PDF includes structured tables and visualizations.
