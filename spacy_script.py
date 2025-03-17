import spacy
import PyPDF2
import pickle
import langchain
import re
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler

###############################################################################
#     Utils
###############################################################################

def load_pdf_text(pdf_path) -> str:
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text



class FinancialDataExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self._add_financial_patterns(type='matcher')


    def _add_financial_patterns(self, type: str ='ruler'):
        """
        Define patterns to match financial entities like 'Total Assets', 'Operating Income', etc.
        """
        assert type in ['ruler', 'matcher'], f"Unsupported type {type}"
        if type == 'ruler':
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            
            financial_terms = [
                {"label": "FINANCIAL_TERM", "pattern": "Total Assets"},
                {"label": "FINANCIAL_TERM", "pattern": "Operating Income"},
                {"label": "FINANCIAL_TERM", "pattern": "Total Liabilities"},
                {"label": "FINANCIAL_TERM", "pattern": "Total Sales"},
                {"label": "FINANCIAL_TERM", "pattern": "Total Revenue"},
                {"label": "FINANCIAL_TERM", "pattern": "Total Expenses"},
                {"label": "FINANCIAL_TERM", "pattern": "Total Transaction Costs"}
            ]
            ruler.add_patterns(financial_terms)
        
        elif type == 'matcher':
            financial_terms = [
                "Total Assets", 
                "Operating Income", 
                "Total Liabilities",
                "Total Sales", 
                "Total revenue", 
                "Total expenses", 
                "Total transaction costs"
            ]

            for term in financial_terms:
                pattern = [{"LOWER": token.lower()} for token in term.split()]
                self.matcher.add(term, [pattern])


    def extract_financial_data(self, text, type: str ='ruler'):
        """
        Extracts financial terms and associated numerical values.
        :param text: str, Financial document text
        :return: dict, Extracted financial values
        """
        assert type in ['ruler', 'matcher'], f"Unsupported type {type}"
        doc = self.nlp(text)
        extracted_data = {}

        if type == 'ruler':
            for ent in doc.ents:
                if ent.label_ == "FINANCIAL_TERM":
                    value = self._find_numeric_value(doc, ent.end)
                    if value:
                        extracted_data[ent.text] = value
        elif type == 'matcher':
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                entity = doc[start:end].text
                value = self._find_numeric_value(doc, end)
                if value:
                    extracted_data[entity] = value

        return extracted_data


    def _find_numeric_value(self, doc, start_index):
        """
        Finds the closest numerical value to the matched financial term.
        :param doc: spaCy Doc object
        :param start_index: int, Index after matched entity
        :return: str, Extracted numerical value
        """
        for token in doc[start_index:]:
            # Match currency and number patterns like "$100M", "USD 5 billion", "4,500,000"
            if re.match(r"(\$|USD)?\s?\d{1,3}(,\d{3})*(\.\d+)?(M|B|million|billion)?", token.text):
                return token.text

        return None

if __name__ == "__main__":
    pdf_path = "data/Blackrock_AR.pdf"
    text = load_pdf_text(pdf_path)

    extractor = FinancialDataExtractor()
    result = extractor.extract_financial_data(text, type='matcher')
    for k, v in result.items():
        print(f'{k}: {v}')



