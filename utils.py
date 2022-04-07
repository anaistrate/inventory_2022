import argparse
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Mapping from generic model name to the Huggingface Version used to initialize the model
MODEL_TO_HUGGINGFACE_VERSION = {
  'bert' : 'bert_base_uncased',
  'biobert' : 'dmis-lab/biobert-v1.1',
  'scibert' : 'allenai/scibert_scivocab_uncased',
  'pubmedbert' : 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
  'pubmedbert_fulltext' : 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
  'bluebert' : 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',
  'bluebert_mimic3' : 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
  'sapbert' : 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
  'sapbert_mean_token' : 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token',
  'bioelectra' : 'kamalkraj/bioelectra-base-discriminator-pubmed',
  'bioelectra_pmc' : 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc',
  'electramed' : 'giacomomiolo/electramed_base_scivocab_1M',
  'biomed_roberta' : 'allenai/biomed_roberta_base',
  'biomed_roberta_chemprot' : 'allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169',
  'biomed_roberta_rct_500' : 'allenai/dsp_roberta_base_dapt_biomed_tapt_rct_500'
}

# Mapping from NER tag to ID
NER_TAG2ID = {'O' : 0, 'B-RES' : 1, 'I-RES' : 2}

# Mapping from ID to NER tag
ID2NER_TAG = {v:k for k, v in NER_TAG2ID.items()}

# Hyperparameters used for training
ARGS_MAP = {
  'bert' : ['bert-base-uncased', 16, 3e-5, 0, False],
  'biobert' : ['dmis-lab/biobert-v1.1', 16, '3e-5', 0, False],
  'scibert' : ['allenai/scibert_scivocab_uncased', 16, 3e-5, 0, False],
  'pubmedbert' : ['microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', 16, 3e-5, 0, True],
  'pubmedbert_fulltext' : ['microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 32, 3e-5, 0, True],
  'bluebert' : ['bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12', 16, 3e-5, 0, True],
  'bluebert_mimic3' : ['bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12', 32, 3e-5, 0, False],
  'sapbert' : ['cambridgeltl/SapBERT-from-PubMedBERT-fulltext', 16, 2e-5, 0.01, False],
  'sapbert_mean_token' : ['cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token', 32, 2e-5, 0.01, False],
  'bioelectra' : ['kamalkraj/bioelectra-base-discriminator-pubmed', 16, 5e-5, 0, True],
  'bioelectra_pmc' : ['kamalkraj/bioelectra-base-discriminator-pubmed-pmc', 32, 5e-5, 0, True],
  'electramed' : ['giacomomiolo/electramed_base_scivocab_1M', 16, 5e-5, 0, True],
  'biomed_roberta' : ['allenai/biomed_roberta_base', 16, 2e-5, 0, False],
  'biomed_roberta_rct500' : ['allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169', 16, 2e-5, 0, False],
  'biomed_roberta_chemprot' : ['allenai/dsp_roberta_base_dapt_biomed_tapt_rct_500', 16, 2e-5, 0, False]
}

def set_random_seed(seed):
  """
  Sets random seed for deterministic outcome of ML-trained models
  """
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def get_parsed_xml(text):
  """
  Strips XML tags from a string
  :param text: string possibly containing XML tags
  :return: string parsed of XML tags
  """
  text_xml_parsed = ""
  substring = text
  while '<' in substring and '</' in substring and '>' in substring:
    start_tag_open = substring.find('<')
    start_tag_close = substring.find('>')

    end_tag_open = substring.find('</')
    end_tag_close = substring.find('>', end_tag_open)

    if start_tag_open != -1 and start_tag_close != -1 and end_tag_open != -1 and end_tag_close != 1:
      parsed_xml = substring[(start_tag_close + 1): end_tag_open]
      text_xml_parsed += substring[:start_tag_open] + parsed_xml + " "
    substring = substring[(end_tag_close + 1):]
  text_xml_parsed += substring
  return text_xml_parsed