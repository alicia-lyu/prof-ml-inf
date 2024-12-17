import ocrmypdf
import time
import fitz 

def extract_all_text(pdf_path):
    global io_time, ocr_time
    # Open the PDF document
    t1 = time.time()
    pdf_document = fitz.open(pdf_path)
    t2 = time.time()
    io_time = 1000 * (t2 - t1)
    all_text = ""

    # Loop through all pages and concatenate text
    t1 = time.time()
    # print(range(len(pdf_document)))
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        all_text += page.get_text() + "\n"  # Add a newline after each page
        if page_number == 6:
            break
        

    t2 = time.time()
    ocr_time = 1000 * (t2 - t1)
    pdf_document.close()
    return all_text

def run_ocr(input_pdf: str, size_threshold: int = 10 * 1024 * 1024) -> str:
    """
    Perform OCR on a PDF. Store the result in memory for small PDFs,
    or write to a file for large PDFs based on a size threshold.

    Args:
        input_pdf (str): Path to the input PDF file.
        size_threshold (int): File size threshold in bytes (default: 10 MB).

    Returns:
        str: The OCR output text.
    """
    # Check the size of the input PDF
    ocrmypdf.ocr(
        input_file=input_pdf,
        output_file="ocr_temp.out",
        skip_text=True
    )
    return extract_all_text("ocr_temp.out")
        
        

if __name__ == "__main__":
    # Example usage:
    input_pdf_path = "pdfs/pipedream.pdf"
    print(extract_all_text(input_pdf_path))

    ocr_text = run_ocr(input_pdf_path)
    print(ocr_text)