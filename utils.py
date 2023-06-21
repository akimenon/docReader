# Import PyPDF4

# convert PDF to text
def textfrompdf(pdfDocument):
    #print(len(pdfDocument.pages))
    # Create an empty list to store documents
    text = ""

    # Iterate over all pages in the PDF
    for page in pdfDocument.pages:
        # Extract text from each page
        text += page.extract_text()

    # Print the list of documents
    return text
