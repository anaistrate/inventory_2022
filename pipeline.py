import requests
import pandas as pd
from paper_classif_predictor import *
from ner_predictor import *
from url_predictor import *
from utils import *

def get_title_abstract(results):
	"""
	Retrieves title and abstracts from the results from querying EuropePMC

	:param results: JSON results of a EuropePMC query

	:return df: df containing PMIDs, titles and abstracts from parsing results
	"""
	papers = results['resultList']['result']
	pmids = []
	titles = []
	abstracts = []
	for i, paper in enumerate(papers):
	  pmid = paper['pmid']
	  if i % 5 == 0:
	  	print('Parsing paper', i, ":", pmid)
	  title = paper['title']
	  if 'abstractText' in paper.keys():
	    abstract = paper['abstractText']
	  else:
	    abstract = None
	  pmids.append(pmid)
	  titles.append(title)
	  abstracts.append(abstract)
	df = pd.DataFrame({'PMID' : pmids, 'title' : titles, 'abstract' : abstracts})
	return df


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--query', type=str, default = 'cancer')

	parser.add_argument('--paper-classif-checkpt', type=str, default = 'checkpts/checkpt_biomed_roberta_title_abstract_512_10_epochs', help = 'Location of saved checkpoint file for Paper Classification task.')
	parser.add_argument('--ner-checkpt', type=str, default = 'checkpts/checkpt_ner_pubmedbert_fulltext_7_epochs', help = 'Location of saved checkpoint file for NER task.')
	parser.add_argument('--paper-classif-model-name', type=str, default='biomed_roberta', help = "Name of Paper Classification model to try. Can be one of: ['bert', 'biobert', 'scibert', 'pubmedbert', 'pubmedbert_pmc', 'bluebert', 'bluebert_mimic3', 'sapbert', 'sapbert_mean_token', 'bioelectra', 'bioelectra_pmc', 'electramed', 'biomed_roberta', 'biomed_roberta_chemprot', 'biomed_roberta_rct_500']")
	parser.add_argument('--ner-model-name', type=str, default='pubmedbert_fulltext', help = "Name of NER model to try. Can be one of: ['bert', 'biobert', 'scibert', 'pubmedbert', 'pubmedbert_pmc', 'bluebert', 'bluebert_mimic3', 'sapbert', 'sapbert_mean_token', 'bioelectra', 'bioelectra_pmc', 'electramed', 'biomed_roberta', 'biomed_roberta_chemprot', 'biomed_roberta_rct_500']")

	parser.add_argument('--paper-classif-output-file', type=str, default = 'output_dir/preds_paper_classifications.csv', help = 'Output file containing Paper Classification predictions')
	parser.add_argument('--ner-output-file', type=str, default = 'output_dir/preds_ner.csv', help = 'Output file containing NER predictions')
	parser.add_argument('--query_europepmc', type=bool, default=False)
	parser.add_argument('--papers-file', type=str, default = 'data/val_paper_classif.csv')
	
	parser.add_argument('--descriptive-labels', type=str, default=['not-bio-resource', 'bio-resource'], help = "Descriptive labels corresponding to the [0, 1] numeric scores")
	parser.add_argument('--predictive-field', type=str, default='title_abstract', help = "Field in the dataframes to use for prediction. Can be one of ['title', 'abstract', 'title-abstract']")
	        
	args, _ = parser.parse_known_args()
	print(f'args={args}')

	if args.query_europepmc:
		# Query EuropePMC
		print('Querying EuropePMC')
		url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="  + args.query + "&resultType=core&fromSearchPost=false&format=json"
		results = requests.get(url).json()
		papers_df = get_title_abstract(results)
		papers_df.to_csv(args.papers_file)
		print('Finished querying EuropePMC. Saved', args.papers_file, 'to file!')

	# Paper Classification Model
	model_huggingface_version = MODEL_TO_HUGGINGFACE_VERSION[args.paper_classif_model_name]
	paper_classif_predictor = PaperClassificationPredictor(model_huggingface_version, args.paper_classif_checkpt, args.descriptive_labels)

	# Load data in a DataLoader
	print('Processing dataset ...')
	data_handler = DataHandler(model_huggingface_version, args.papers_file, pred_only = True)
	data_handler.parse_abstracts_xml()
	data_handler.concatenate_title_abstracts()
	print('Finished processing dataset!')

	# Predict labels
	print('Starting prediction ...')
	predicted_labels = []
	sentences = data_handler.train_df[args.predictive_field].values
	print('Generating predictions for ', len(sentences), 'sentences!')
	predictor = PaperClassificationPredictor(model_huggingface_version, args.paper_classif_checkpt, args.descriptive_labels)
	for sentence in sentences:
		predicted_label = predictor.predict(sentence)[0]
		predicted_labels.append(predicted_label)
	data_handler.train_df['predicted_label'] = predicted_labels
	pred_df = data_handler.train_df
	if 'curation_score' in data_handler.train_df.columns:
		true_labels = [paper_classif_predictor.class_labels.int2str(int(x)) for x in data_handler.train_df['curation_score'].values]
		pred_df['true_label'] = true_labels
		pred_df = data_handler.train_df
	print('Finished paper classification prediction!')
	print(pred_df[:20])

	# Save labels to file
	pred_df.to_csv(args.paper_classif_output_file)
	print('Saved predictions to', args.paper_classif_output_file)


	# Predict labels for papers 
	model_huggingface_version = MODEL_TO_HUGGINGFACE_VERSION[args.ner_model_name]
	ner_predictor = NERPredictor(model_huggingface_version, args.ner_checkpt)
	url_predictor = URLPredictor()
	
	all_pred_databases = []
	all_IDs = []
	all_texts = []
	all_urls = []
	all_paper_labels = []
	all_probabilities = []
	all_offsets_start = []
	all_offsets_end = []

	id_field = 'PMID'
	if id_field not in pred_df.columns.values:
		id_field = 'id'
	papers_biodata_res = pred_df[pred_df['predicted_label'] == 'bio-resource'][id_field].values
	unique_papers_biodata_res = set(papers_biodata_res)
	
	IDs = pred_df[id_field].values
	text_arr = pred_df[args.predictive_field]
	paper_labels = pred_df['predicted_label']

	paper_labels = pred_df['predicted_label'].values
	for ID, text, paper_label in zip(IDs, text_arr, paper_labels):
		if ID in unique_papers_biodata_res:
			predicted_labels = ner_predictor.predict(text)
			predicted_urls = list(url_predictor.get_urls(text))
			databases = [x['word'] for x in predicted_labels]
			probabilities = [x['prob'] for x in predicted_labels]
			offsets_start = [x['start'] for x in predicted_labels]
			offsets_end = [x['end'] for x in predicted_labels]
		else:
			predicted_labels = [None]
			predicted_urls = [None]
			databases = [None]
			probabilities = [None]
			offsets_start = [None]
			offsets_end = [None]
		num_preds = len(predicted_labels)
		all_pred_databases.extend(databases)
		all_probabilities.extend(probabilities)
		all_offsets_start.extend(offsets_start)
		all_offsets_end.extend(offsets_end)
		all_IDs.extend([ID] * num_preds)
		all_texts.extend([text] * num_preds)
		all_urls.extend([predicted_urls] * num_preds)
		all_paper_labels.extend([paper_label] * num_preds)
	ner_pred_df = pd.DataFrame({id_field : all_IDs, 'predicted_label' : all_paper_labels, 'text' : all_texts, 'database' : all_pred_databases, 'probability' : all_probabilities, 'start_offset' : all_offsets_start, 'end_offset' : all_offsets_end, 'URL' : all_urls})
	print(ner_pred_df[:20])

	# Save labels to file
	ner_pred_df.to_csv(args.ner_output_file)
	print('Saved NER and URL predictions to', args.ner_output_file)

