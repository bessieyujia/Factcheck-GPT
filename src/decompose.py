import nltk
# nltk.download()
# from nltk import sent_tokenize
# import spacy
# nlp = spacy.load("en_core_web_sm")
from utils.openaiAPI import gpt
from utils.data_util import save_to_file
from utils.prompt import DOC_TO_INDEPEDENT_SENTENCES_PROMPT, SENTENCES_TO_CLAIMS_PROMPT, DOC_TO_SENTENCES_PROMPT
from typing import List


def doc_to_sents(text: str, tool_name = "nltk") -> List[str]:
    if tool_name == "nltk":
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip())>=3]
    # elif tool_name == "spacy":
    #     doc = nlp(text)
    #     sentences = [str(sent).strip() for sent in doc.sents]
    return sentences


def doc2sentences(doc: str, mode: str="independent_sentences",
                  model: str="gpt-3.5-turbo", 
                  system_role: str="You are good at decomposing and decontextualizing text.",
                  num_retries: int=3) -> List[str]:
    if mode == "sentences":
        prompt = DOC_TO_SENTENCES_PROMPT
    elif mode == "independent_sentences":
        prompt = DOC_TO_INDEPEDENT_SENTENCES_PROMPT
    elif mode == "claims":
        prompt = SENTENCES_TO_CLAIMS_PROMPT

    results = None
    user_input = prompt.format(doc=doc).strip()

    r = None  # Ensure r is defined
    for _ in range(num_retries):
        r = gpt(user_input, model=model, system_role=system_role)
        print(r)
        try:
            results = eval(r) if r else None  # Only eval if r is valid
            print(results)
            if isinstance(results, list):  # Break if result is as expected
                return results
        except Exception as e:
            print(f"Error in doc2sentences processing: {e}")
    
    # Fallback to NLTK tokenization if retries fail or result is incorrect
    print(f"Model output: {r}. Using NLTK sentence split as fallback.")
    sentences = nltk.sent_tokenize(doc)
    return [s.strip() for s in sentences if len(s.strip()) >= 3]


    # for _ in range(num_retries):
    #     try:
    #         r = gpt(user_input, model=model, system_role=system_role)
    #         results = eval(r) if r else None  # Avoid eval if r is None
    #         if isinstance(results, list):  # Break if successful
    #             break
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}.")
    #         if r:  # Only save if r has a value
    #             save_to_file(r)
    #         else:
    #             print("No output to save.")
    
    # # Fallback if `results` is not a list
    # if not isinstance(results, list):
    #     print(f"{model} output {r}. It does not output a list of sentences correctly, returning NLTK split results.")
    #     return doc_to_sents(doc, tool_name="nltk")
    
    # return results


    # original.....
    # for _ in range(num_retries):
    #     try:
    #         r = gpt(user_input, model=model, system_role=system_role)
    #         results = eval(r)
    #         break
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}.")
    #         save_to_file(r)

    # if isinstance(results, list):
    #     return results
    # else:
    #     print(f"{model} output {r}. It does not output a list of sentences correctly, return NLTK split results.")
    #     return doc_to_sents(doc, tool_name = "nltk")