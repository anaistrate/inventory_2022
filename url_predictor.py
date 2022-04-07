import pandas as pd
import re
import string
import argparse
import nltk
from utils import *
nltk.download('punkt')

class URLPredictor():
  """
  Handles Prediction URLs
  """
  
  def __init__(self, data_file = None, predict_only = None):
    """
    :param data_file: data file to extract URLs from; could be none if the URLPredictor is used directly on text
    :param predict_only: true if only used for prediction
    """
    if not data_file:
      return
    df = pd.read_csv(data_file)
    if predict_only:
      df = df[['id', 'title', 'abstract']]
    else:
      df = df[['id', 'title', 'abstract', 'url']]
    df['url_true'] = df['url'].apply(lambda x: str(x).strip(string.punctuation))
    df = df.groupby(['id', 'title', 'abstract']).agg(set).reset_index()
    df['url_true'] = df['url_true'].apply(list)
    df['abstract_parsed_xml'] = df['abstract'].apply(get_parsed_xml) 
    df['title_parsed_xml'] = df['title'].apply(get_parsed_xml) 
    self.df = df

  def concatenate_predictions(self, x):
    """
    Returns the union of predicted URLs from both titles and abstracts
    """
    return list(x['url_pred_title'].union(x['url_pred_abstract']))

  def get_urls(self, text):
    """
    Returns a set of URls from given text, using a regular expression

    :param text: text to retrieve URLs from

    :return: list of predicted URLs
    """
    pred_urls = []
    offsets = []
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
      url_start = sent.find('http')
      while url_start < len(sent) and url_start != -1:
        url_end = sent.find(' ', url_start)
        url = sent[url_start:url_end]
        pred_urls.append(url.strip(string.punctuation))
        offsets.append((url_start, url_end))
        url_start = sent.find('http', url_end)
    pred_urls = set(pred_urls)
    return pred_urls

  def predict_urls(self):
    """
    Retrieves all URLs from the title and abstracts present in the initial df
    """
    self.df['url_pred_title'] = self.df['title_parsed_xml'].apply(lambda x: self.get_urls(x))
    self.df['url_pred_abstract'] = self.df['abstract_parsed_xml'].apply(lambda x: self.get_urls(x))
    self.df['url_pred'] = self.df.apply(self.concatenate_predictions, axis = 1)
    self.df = self.df.drop(columns = ['url_pred_title', 'url_pred_abstract'])
    return self.df
  
  def get_metrics(self):
    """
    Computes metrics for prediction method for URLs from title/abstract fields

    :return: precision, recall, f1 score
    """
    all_true_urls = self.df['url_true'].values
    all_pred_urls = self.df['url_pred'].values
    tp = 0
    fp = 0
    fn = 0
    false_positives = []
    false_negatives = []
    for true_urls, pred_urls in zip(all_true_urls, all_pred_urls):
      for pred in pred_urls:
        if pred in true_urls:
          tp += 1
        else:
          fp += 1
          false_positives.append(pred)
      for true in true_urls:
        if true not in pred_urls:
          fn += 1
          false_negatives.append(true)

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print('Precision: {:0.4f}, Recall: {:0.4f}, F1: {:0.4f}'.format(p, r, f1))
    return p, r, f1


if __name__ == '__main__':
  # Parsing arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file', type=str, default='data/extracted_elements_2022-02-24_hji.csv', help = 'Location of input file')
  parser.add_argument('--output-file', type=str, default='output_dir/preds_urls.csv', help = 'Location of output file')
  parser.add_argument('--predict-only', type=bool, default = False, help = 'True if to predict only. Input file should have ["id", "abstract", "title"] fields. Otherwise it is assumed that input file has ["id", "abstract", "title", "url"] fields')

  args, _ = parser.parse_known_args()

  url_predictor = URLPredictor(args.input_file, args.predict_only)
  pred_df = url_predictor.predict_urls()
  pred_df.to_csv(args.output_file)
  if not args.predict_only:
    metrics = url_predictor.get_metrics()
  
  #Examples of extracting URLs from two pieces of text
  text1 = "Information from structural genomics experiments at the RIKEN SPring-8 Center, Japan has been compiled and published as an integrated database. The contents of the database are (i) experimental data from nine species of bacteria that cover a large variety of protein molecules in terms of both evolution and properties (http://database.riken.jp/db/bacpedia), (ii) experimental data from mutant proteins that were designed systematically to study the influence of mutations on the diffraction quality of protein crystals (http://database.riken.jp/db/bacpedia) and (iii) experimental data from heavy-atom-labelled proteins from the heavy-atom database HATODAS (http://database.riken.jp/db/hatodas). The database integration adopts the semantic web, which is suitable for data reuse and automatic processing, thereby allowing batch downloads of full data and data reconstruction to produce new databases. In addition, to enhance the use of data (i) and (ii) by general researchers in biosciences, a comprehensible user interface, Bacpedia (http://bacpedia.harima.riken.jp), has been developed."

  text2 = "DPL (http://www.peptide-ligand.cn/) is a comprehensive database of peptide ligand (DPL). DPL1.0 holds 1044 peptide ligand entries and provides references for the study of the polypeptide platform. The data were collected from PubMed-NCBI, PDB, APD3, CAMPR3, etc. The lengths of the base sequences are varied from 3 to78. DPL database has 923 linear peptides and 88 cyclic peptides. The functions of peptides collected by DPL are very wide. It includes 540 entries of antiviral peptides (including SARS-CoV-2), 55 entries of signal peptides, 48 entries of protease inhibitors, 45 entries of anti-hypertension, 37 entries of anticancer peptides, etc. There are 270 different kinds of peptide targets. All peptides in DPL have clear binding targets. Most of the peptides and receptors have 3D structures experimentally verified or predicted by CYCLOPS, I-TASSER and SWISS-MODEL. With the rapid development of the COVID-2019 epidemic, this database also collects the research progress of peptides against coronavirus. In conclusion, DPL is a unique resource, which allows users easily to explore the targets, different structures as well as properties of peptides."
  
  pred_urls1 = url_predictor.get_urls(text1)
  print("Text: ", text1)
  print("=" * 30)
  print("Pred URLs: ", pred_urls1)
  print()
  
  pred_urls2 = url_predictor.get_urls(text2)
  print("Text: ", text2)
  print("=" * 30)
  print("Pred URLs: ", pred_urls2)