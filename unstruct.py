from unstructured.partition.pdf import partition_pdf
import re
import json

if __name__ == "__main__":
    with open('config.json', 'r') as config_f:
        config = json.load(config_f)
        config = config[config['mode']]


    # extract text
    text_elements = partition_pdf(config['file'])
    text = "\n".join([str(el) for el in text_elements])

    # Define regex patterns for financial data extraction
    patterns = {
        "Total Assets": r"(?i)total assets[\s:]*([\d,\.]+)",
        "Operating Income": r"(?i)operating income[\s:]*([\d,\.]+)",
        "Total Liabilities": r"(?i)total liabilities[\s:]*([\d,\.]+)",
        "Total Sales": r"(?i)Total sales[\s:]*([\d,\.]+)",
        "Total purchases": r"(?i)Total purchases[\s:]*([\d,\.]+)",
        "Total assets": r"(?i)total assets[\s:]*([\d,\.]+)",
        "Total revenue": r"(?i)total revenue[\s:]*([\d,\.]+)",
        "Total expenses": r"(?i)total expenses[\s:]*([\d,\.]+)",
        "Total transaction costs": r"(?i)total transaction costs[\s:]*([\d,\.]+)",
        "Total transaction costs in 2023": r"(?i)total transaction costs 2023[\s:]*([\d,\.]+)",
    }

    # Extract financial data
    extracted_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            extracted_data[key] = match.group(1)
        else:
            extracted_data[key] = "Not Found"

    # Print extracted financial data
    for key, value in extracted_data.items():
        print(f"{key}: {value}")
