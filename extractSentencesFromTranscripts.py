import csv
import getopt
import itertools
import os
import re
import sys
from pathlib import Path
from nltk.tokenize import sent_tokenize
from lib.SemanticSentenceMatcher import SemanticMatcher, current_milli_time, TestSequence, TrainingSequence
from spellchecker import SpellChecker
from lib.spellcheck.typo_dict import typo_dict, pattern_dict, name_list, medicine_list, common_phrase_list, parse_int

INPUT_CONSOLE = 'console'
INPUT_FILE = 'file'


def getKeyListForSortedClusters(clustered_sentences):
    keyList = sorted(clustered_sentences, key=lambda k: len(clustered_sentences[k]), reverse=True)
    return keyList


def printer(matched_results, output_filepath=None):
    if output_filepath is None:
        for match in matched_results:
            print(match)
    else:
        print("Writing output at:", output_filepath)
        with open(output_filepath, 'a') as ofp:
            # ofp.write(json.dumps(text))
            for match in matched_results:
                str_match = str(match)
                ofp.write(str_match)
                if "\n" not in str_match:
                    ofp.write("\n")


class MainHandler(object):
    def __init__(self):
        self.USAGE = f"Usage: python {sys.argv[0]} [--help | -h] | [--version | -v] | " \
                     f"[--input_dir | -i =<path to the dir containing the corpus>] | " \
                     f"[--output_dir | -o =<path to the dir where the output should be written>] | " \
                     f"[--output_filename | -n =<name of the output file>] | " \
                     f"[--pattern_filepath | -p =<path to the labeled pattern file>]" \
                     f"[--labeled_seq | -l =<path to the labeled sequence file>]" \
                     f"[--max_sim | -m]" \
                     f"[--seq_only | -s]"\
                     f"[--cross | -c Only works with the -s|--seq_only option to produce a combination of the sentences]"

        self.version = 2.0
        self.training_model_path = 'Embeddings/*'
        # self.semMatcher = SemanticMatcher("model/distilbert-base-uncased_bal_const_12e_minlr_32b")
        # self.semMatcher = SemanticMatcher("all-mpnet")
        self.semMatcher = SemanticMatcher("custom")
        self.spellchecker = SpellChecker()
        self.sequence_labels = dict()
        # Mean Pooling - Take attention mask into account for correct averaging

    def createEmbeddingsForPatterns(self, pattern_list):
        pattern_embeddings = []
        for patt in pattern_list:
            patt_items = patt.split(";;")
            try:
                _emb_patt = TrainingSequence(patt_items[0], patt_items[1], patt_items[2])
            except IndexError:
                print(patt_items[0])
                print(patt_items[1])
                print(patt_items[2])
                sys.exit(5)
            _emb_patt.embedding_vector = self.semMatcher.getEmbeddingVector(patt_items[0])
            pattern_embeddings.append(_emb_patt)
        return pattern_embeddings

    def createEmbeddingsForSequences(self, corpus):
        test_sequence_list = []
        for c in corpus:
            if len(str(c).strip()) < 1:
                continue
            if ";;" in c:
                items = c.split(";;")
                if len(items) == 2:
                    t = TestSequence(items[0], items[1])
                elif len(items) == 3:
                    t = TestSequence(items[0], items[1], items[2])
                else:
                    t = TestSequence(items[0])

                t.embedding_vector = self.semMatcher.getEmbeddingVector(t.sentence)
            else:
                t = TestSequence(c)
                t.embedding_vector = self.semMatcher.getEmbeddingVector(t.sentence)
            test_sequence_list.append(t)
        return test_sequence_list

    def clean_text(self, text):
        str_sent = str(text)
        str_sent = "'".join([x for x in re.split("\"|\'", str_sent) if x != ''])
        if ":" in str_sent:
            str_sent = str_sent.rsplit(':', 1)[1]
        # Fixing spaces and spellings
        misspelled_words = self.spellchecker.unknown(
            str_sent
                # .replace(":", " ")
                .replace(",", " ")
                .replace(".", " ")
                .replace("?", " ")
                .split())

        #convert textual numbers to number
        str_sent = parse_int(str_sent)
        for mw in misspelled_words:
            if mw not in name_list and mw not in medicine_list and mw not in common_phrase_list:
                if mw in typo_dict:
                    str_sent = str_sent.replace(mw, typo_dict[mw])
                #else:
                    #print(mw, "-??-", self.spellchecker.candidates(mw))

        for pat_key, pat_value in pattern_dict.items():
            if pat_key in str_sent:
                str_sent = ". ".join(str_sent.split(pat_key))

        str_sent.replace("?.", "?")

        # print(str_sent)
        str_sent = " ".join(str_sent.split())
        str_sent = str_sent.strip(".")

        return str_sent

    def main(self):
        args = sys.argv[1:]
        input_dir = "Data/transcripts/processed/"
        pattern_filepath = ""
        input_method = INPUT_FILE
        isMaxSimilarityOnly = False
        isSequenceOnly =False
        labeled_sequence_filepath = ""
        _str_output_filename=""
        isCross = False
        _min_sim = 0.1
        _str_output_dir = 'matching/transcripts/'
        read_file_extensions = ("*.txt", "*.csv")
        try:
            options, arguments = getopt.getopt(args, "hvxsci:p:l:o:n:m:", ["help", "version", "max_sim", "seq_only", "cross",
                                                                  "input_dir=", "pattern_filepath=", "labeled_seq=",
                                                                     "output_dir=", "output_filename=", "min_sim"])
        except getopt.GetoptError:
            print(self.USAGE)
            sys.exit()
        for o, a in options:
            if o in ("-h", "--help"):
                print(self.USAGE)
                sys.exit()
            if o in ("-v", "--version"):
                print(self.version)
                sys.exit()
            if o in ("-p", "--pattern_filepath"):
                pattern_filepath = a
            if o in ("-i", "--input_dir"):
                input_dir = a
            if o in ("-o", "--output_dir"):
                _str_output_dir = a
            if o in ("-n", "--output_filename"):
                _str_output_filename = a
            if o in ("-x", "--max_sim"):
                isMaxSimilarityOnly = True
            if o in ("-m", "--min_sim"):
                _min_sim = float(a)
            if o in ("-s", "--seq_only"):
                isSequenceOnly = True
                if o in ("-c", "--cross"):
                    isCross = True
            if o in ("-l", "--labeled_seq"):
                labeled_sequence_filepath = a


        if input_method == "":
            print('No input methods were supplied')
            print(self.USAGE)
            sys.exit()

        input_dir_handle = Path(input_dir)
        sentence_list = []
        output_dir = Path(_str_output_dir)
        if not output_dir.is_dir():
            print("output dir does not exists, creating a new one.")
            os.makedirs(output_dir)

        if _str_output_filename == "":
            if isSequenceOnly:
                output_file = "transcript_sequences" + str(self.version) + "_" + str(current_milli_time()) + ".csv"
            else:
                output_file = "transcript_vectors" + str(self.version) + "_" + str(current_milli_time()) + ".csv"
        else:
            output_file = _str_output_filename

        if isCross:
            output_cross_file = "transcript_sequence_cross" + str(self.version) + "_" + str(
                current_milli_time()) + ".csv"

        output_filepath = output_dir.joinpath(output_file)
        if not isSequenceOnly:
            if not pattern_filepath:
                print('No pattern filepath was supplied')
                print(self.USAGE)
                sys.exit()
            pattern_file = Path(pattern_filepath)
            pattern_list = []

            if not pattern_file.is_file():
                print('No pattern file was supplied')
                print(self.USAGE)
                sys.exit()

            # # Load Training data from EMR embeddings
            # print('Loading EMR Training Data...........', end="")
            # self.semMatcher.loadTrainingData(self.training_model_path)
            # print('Complete')

            with pattern_file.open() as items:
                for item in items:
                    if len(item) < 1:
                        continue
                    if item.startswith("*"):
                        item = item.strip("*")
                        extraItems = [item]
                        # if item.startswith("the patient's ") and "%x" in item:
                        #     extraItems.append(item.replace("the patient's ", "his ", 1).replace("the patient ", "he "))
                        #     extraItems.append(item.replace("the patient's ", "her ", 1).replace("the patient ", "she "))
                        # elif item.startswith("the patient ") and "%x" in item:
                        #     extraItems.append(item.replace("the patient ", "he ", 1))
                        #     extraItems.append(item.replace("the patient ", "she ", 1))
                        # elif item.startswith("patient ") and "%x" in item:
                        #     extraItems.append(item.replace("patient ", "he ", 1))
                        #     extraItems.append(item.replace("patient ", "she ", 1))
                        pattern_list.extend(extraItems)
            self.semMatcher.trainedData.extend(self.createEmbeddingsForPatterns(pattern_list))

            print("len(self.semMatcher.trainedData):", len(self.semMatcher.trainedData))
        else:
            if not labeled_sequence_filepath:
                print('No labeled sequence file has been provided. Will fill with default similarity of 0')
            else:
                labeled_sequence_file = Path(labeled_sequence_filepath)
                self.sequence_labels = dict()
                with labeled_sequence_file.open() as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter='|')
                    line_count = 0
                    for row in csv_reader:
                        if line_count == 0:
                            print(f'Column names are {", ".join(row)}')
                            line_count += 1
                        else:
                            if len(row)<3:
                                print(row)
                                continue
                            # _ls = LabeledSequence(row[1],row[2],row[3])
                            _sentence_left = str(row[1]).strip()
                            _sentence_right = row[2].strip()
                            _label = row[3].strip()
                            if _sentence_left not in self.sequence_labels.keys():
                                self.sequence_labels[_sentence_left] = dict()
                            self.sequence_labels[_sentence_left][_sentence_right] = _label
                            line_count += 1
                    print(f'Processed {line_count} lines.')

        if input_dir_handle.is_dir():
            files = [f for f in input_dir_handle.iterdir() if any(f.match(p) for p in read_file_extensions)]
            print("Found ", len(files)," files in the input directory")
            for child in files:
                if child.is_file():
                    with child.open() as testItems:
                        for item in testItems:
                            if len(item) < 1:
                                continue

                            if re.search("(\w+\s+\.\s*\w+)||(\w+\.\w+)", item):  # " ." in item:
                                item = re.sub(r"\.(([^\w]+)?)", r". \1", item)
                                item = re.sub(r"(([^\w])?)\s+\.", r'\1.', item)
                            if re.search("(\w+\s+\?\s*\w+)|(\w+\?\w+)", item):
                                item = re.sub(r"\?(([^\w]+)?)", r"? \1", item)
                                item = re.sub(r"(([^\w])?)\s+\?", r'\1?', item)
                            if re.search("(\w+\s+:\s*\w+)|(\w+:\w+)", item):  # " :" in item:
                                item = re.sub(r":(([^\w]+)?)", r": \1", item)
                                item = re.sub(r"(([^\w])?)\s+:", r'\1:', item)
                            if re.search("and\s*(a)?\s+half", item):  # " :" in item:
                                item = re.sub(r"and\s*(a)?\s+half", "and_a_half", item)

                            # item = re.sub(r"\s+", r"\s", item)
                            item = " ".join(item.split())
                            sentences = sent_tokenize(item)
                            i = 0
                            while i < len(sentences):
                                str_sent = self.clean_text(sentences[i])
                                if "How long have you been she coughing?" in str_sent:
                                    print("HERE 2")
                                    print(sentences[i+1])
                                    exit()
                                str_sent_next = ""

                                if ", " in str_sent:
                                    comma_sep_sent_list = str_sent.split(", ")
                                    for c in comma_sep_sent_list:
                                        if len(c.strip().split(" ")) > 1:
                                            sentence_list.append(c)
                                        if " and " in c:
                                            and_sep_sent_list = c.split(" and ")
                                            for a in and_sep_sent_list:
                                                if len(a.strip().split(" ")) > 1:
                                                    sentence_list.append(a)
                                else:
                                    if " and " in str_sent:
                                        and_sep_sent_list = str_sent.split(" and ")
                                        for a in and_sep_sent_list:
                                            if len(a.strip().split(" ")) > 1:
                                                sentence_list.append(a)

                                if "and_a_half" in str_sent:
                                    str_sent.replace("and_a_half", "and a half")
                                # print("1:",str_sent)
                                if "?" in str_sent and i < len(sentences) - 1:
                                    str_sent_next = self.clean_text(sentences[i + 1])
                                    # print("2:",str_sent_next)
                                    if ", " in str_sent_next:
                                        comma_sep_sent_list = str_sent_next.split(", ")
                                        for c in comma_sep_sent_list:
                                            if len(c.strip().split(" ")) > 1:
                                                sentence_list.append(c)
                                            if " and " in c:
                                                and_sep_sent_list = c.split(" and ")
                                                for a in and_sep_sent_list:
                                                    if len(a.strip().split(" ")) > 1:
                                                        sentence_list.append(a)
                                    else:
                                        if " and " in str_sent_next:
                                            and_sep_sent_list = str_sent_next.split(" and ")
                                            for a in and_sep_sent_list:
                                                if len(a.strip().split(" ")) > 1:
                                                    sentence_list.append(a)

                                    if "and_a_half" in str_sent_next:
                                        str_sent_next.replace("and_a_half", "and a half")
                                        # elif (re.search(r"^[a-zA-Z]+\s+[+-]?((\d+))$", c)):
                                        #     sentence_list.append(c + ";;vital;;%s1 %v1 <Long>")
                                        # elif(re.search(r"^[a-zA-Z]+\s+[+-]?((\d+(\.\d+)?)|(\.\d+))$", c)):
                                        #     sentence_list.append(c+";;vital;;%s1 %v1 <Decimal>")
                                # Check if the seq is a question type, so we can add probable answer after it
                                if str_sent not in sentence_list:
                                    if "?" in str_sent and i < len(sentences) - 1:
                                        # print("2:",str_sent + " " + str_sent_next)
                                        sentence_list.append(str_sent + " " + str_sent_next)
                                        i += 1
                                    else:
                                        # avoid short sentences
                                        if len(str_sent.split()) > 2:
                                            sentence_list.append(str_sent)
                                i += 1
        elif input_dir_handle.is_file():
            with input_dir_handle.open() as testItems:
                for item in testItems:
                    if len(item) < 1:
                        continue
                    if not item.startswith("*") and not item.startswith("----"):
                        item = item.strip()
                        sentence_list.append(item)
        else:
            print("INPUT DIR is not processable")
            print("the handle:", input_dir_handle)
            print("is directory?",input_dir_handle.is_dir())
            print("is file?", input_dir_handle.is_file())
            exit()

        print("len(sentence_list):", len(sentence_list))

        # Removing sentences with pattern matching above some threshold

        if isSequenceOnly:
            printer(sentence_list, output_filepath)
        elif isCross:
            output_cross_filepath = output_dir.joinpath(output_cross_file)
            cross_list = []
            gen = list(itertools.combinations_with_replacement(sentence_list, 2))
            cross_idx = 0
            for u, v in gen:
                if str(u) != str(v):
                    label = str(0)
                    if u in self.sequence_labels.keys():
                        if v in self.sequence_labels[u].keys():
                            label = self.sequence_labels[u][v]

                    cross_list.append(str(cross_idx) + ';;' + str(u) + ";;" + str(v) + ";;" + label+";;")
                    cross_idx += 1
            printer(cross_list, output_cross_filepath)
        else:
            test_seq_list = self.createEmbeddingsForSequences(sentence_list)

            matched_sequence_list = []
            save_step = 10000000
            for _seq_ite, _sequence in enumerate(test_seq_list):
                matching_result = None
                if isMaxSimilarityOnly:
                    matching_result = self.semMatcher.match(_sequence, min_sim=_min_sim)
                    # print(_seq_ite, "= Adding ", len(matching_result), " in previous matched_sequence_list of size: ",
                    #       len(matched_sequence_list))
                    matched_sequence_list.extend(matching_result)
                    # matching_result = self.semMatcher.match(_sequence)
                    # matched_sequence_list.append(matching_result)
                    # print("matching_result:", matching_result)
                else:

                    matching_result = self.semMatcher.match_all_trainingset(_sequence, min_sim=_min_sim)
                    # print(_seq_ite, "= Adding ", len(matching_result), " in previous matched_sequence_list of size: ",
                    #       len(matched_sequence_list))
                    matched_sequence_list.extend(matching_result)
                if len(matched_sequence_list)>save_step:
                    print("Saving len(matched_sequence_list):", len(matched_sequence_list))
                    printer(matched_sequence_list, output_filepath)
                    matched_sequence_list = []

            print("len(matched_sequence_list):", len(matched_sequence_list))
            print("min_sim=", _min_sim)

            printer(matched_sequence_list, output_filepath)

            # printer(self.createEmbeddingsForSequences(sentence_list), output_filepath)
            # print_similarity(sim_result_list)


if __name__ == "__main__":
    mh = MainHandler()
    mh.main()
