import spacy
import PyPDF2
import pickle
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler

###############################################################################
#     Utils
###############################################################################

# Define financial entity patterns
def add_financial_entities(nlp):
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "MONEY", "pattern": "USD"},
        {"label": "MONEY", "pattern": "EUR"},
        {"label": "MONEY", "pattern": "GBP"},
        {"label": "ORG", "pattern": "Barclays"},
        {"label": "FINANCIAL_TERM", "pattern": "interest rate"},
        {"label": "FINANCIAL_TERM", "pattern": "inflation"},
        {"label": "FINANCIAL_TERM", "pattern": "bond yield"},
    ]
    ruler.add_patterns(patterns)

def load_pdf_text(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def extract_financial_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to find financial information based on keywords
def query_financial_text(text, query):
    doc = nlp(text)
    query_doc = nlp(query)
    
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": token.text.lower()} for token in query_doc]
    matcher.add("QUERY_PATTERN", [pattern])
    
    matches = matcher(doc)
    
    results = []
    for match_id, start, end in matches:
        span = doc[start:end].sent
        results.append(span.text)
    
    return results if results else "No relevant information found."

def get_simple_doc():
    return """
        Barclays reported an increase in bond yields this quarter. The interest rate
        was adjusted to 5.5% due to rising inflation. Investors are closely monitoring
        these financial indicators to make informed decisions. USD has strengthened
        against EUR in recent trading sessions.
    """

def run_queries(document, queries):
    output = {}

    # Extract financial entities
    entities = extract_financial_entities(document)
    output['entities'] = entities

    # Query financial document
    responses = []
    for q in queries:
        responses.append(query_financial_text(document, q))

    for i, resp in enumerate(responses):
        print(f"Query Response {i+1}:", resp)

    output['responses'] = responses

    return output


if __name__ == "__main__":

    # Load pretrained spaCy model
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2320500
    
    # Add custom financial entities
    add_financial_entities(nlp)

    queries = ["interest rate", "bond yield"]
    pdf_path = "data/2024_annual_report.pdf"
    annual_report = load_pdf_text(pdf_path)

    simple_out = run_queries(document=get_simple_doc(), queries=queries)
    with open('data/simple_out.pickle', 'wb') as handle:
        pickle.dump(simple_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    annual_report_out = run_queries(document=annual_report, queries=queries)    
    with open('data/annual_report_out.pickle', 'wb') as handle:
        pickle.dump(annual_report_out, handle, protocol=pickle.HIGHEST_PROTOCOL)



