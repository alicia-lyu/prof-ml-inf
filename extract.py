import fitz  # PyMuPDF

def extract_all_text(pdf_path):
    # Open the PDF document
    pdf_document = fitz.open(pdf_path)
    all_text = ""

    # Loop through all pages and concatenate text
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        all_text += page.get_text() + "\n"  # Add a newline after each page

    pdf_document.close()
    return all_text

# Usage
pdf_path = "pdfs/pipedream.pdf"
full_text = extract_all_text(pdf_path)
print(full_text)