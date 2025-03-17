## Purpose

Build an automated pipeline for extraction of data from financial statements.

E.g.
Query: ```What is the fund's total assets under management?```
Response: ```The fund's AUM is $50.3m```

## Files
<ul>
  <li>```llm.py``` - LLM approach for financial data extraction</li>
  <li>```spacy.py``` - rule-based NER approach leveraging spacy ```en_core_web_sm``` pipeline</li>
  <li>```unstruct.py``` - reg-ex based extraction of financial data using predefined patterns </li>  
</ul>
