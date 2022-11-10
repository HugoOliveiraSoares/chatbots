import wikipedia as wiki
import re 

wiki.set_lang("pt")
breast_cancer_content = wiki.page("Breast_cancer").content

classes = re.findall('=\s(.*)\s=',breast_cancer_content)
sentences = re.sub(r'=(.*)=', '',breast_cancer_content)
sentences = re.split(r'\n', sentences)
while '' in sentences:
    sentences.remove('')

print(sentences)
