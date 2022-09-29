# ClinicalConversations

This data repository contains, text obtained by manually transcribing and translating clinical conversation between physicians and patients/guardians.
The data was collected from 2 hospitals in Pakistan. In order to preserve the privacy of the participants, only the anonymized and processed data
is available. 

Some of the ways to execute the associated script are as follows:

1. Generate Sequence

This command will read the data from the Data/raw directory, and produce sequences from it in the matching/transcripts directory.

* every file (with extension txt and csv) will be parsed in the directory specified with -i
* -s is used to generate the sequence files only
* Get the output at matching/transcripts/{transcriptfile}.csv

> python3 extractSentencesFromTranscripts.py -i Data/raw/ -s

2. Classify instances with MASS (training data)

> python3 extractSentencesFromTranscripts.py -i Data/TestData/transcript* -p Data/TrainingData/transcript_annotated_7.csv

* -i can also be used to pass a specific file
* -p is used to provide the training file.
* all lines starting with '*' will be used to create MASS at run time
* The model used here is Fine-Tuned DistilBERT base uncased, which is trained on a portion of un-anonymized data from these conversations.
* Alternatively, all-mpnet or other sentence similarity models can be used by updating this script and the library at lib/SemanticSentenceMatcher.py

> python3 extractSentencesFromTranscripts.py -i Data/TestData/transcript* -p Data/TrainingData/transcript_annotated_7.csv -m 0.87 -x

* -m provides a minimum similarity threshold
* -x is used to select one or more instances with highest similarity. This means, if two instances have a similarity of 0.9, which is the highest (above the specified threshold of 0.87), only these two will be selected.

