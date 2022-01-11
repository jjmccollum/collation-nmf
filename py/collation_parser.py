#!/usr/bin/env python3

import time # to time calculations for users
import re # for parsing augmented witness sigla
from lxml import etree as et # for reading TEI XML inputs
import urllib.request # for making HTTP requests to the VMR API
import numpy as np # matrix support
from sklearn.feature_extraction.text import TfidfTransformer # for reweighting the entries of the input collation matrix to isolated better-separated clusters
from common import * # import all variables from the common support module

"""
Base class for reading collation data and reformatting it as a matrix according to customizable rules.
"""
class collation_parser():
    """
	Constructs a new collation_parser with the given settings.
	"""
    def __init__(self, min_extant_proportion=0.95, use_tfidf=False, ambiguous_rdg_prefix="", subwitness_suffixes=[], trivial_reading_types=[], ignored_reading_types=[], verbose=False):
        self.min_extant_proportion = min_extant_proportion # minumum proportion of variation units where a witness is required to have a reading for that witness to be included in the primary collation matrix
        self.min_extant = 0 # minimum number of ones (i.e., extant readings) required for a column (i.e., a witness) to be included in the primary collation matrix
        self.use_tfidf = use_tfidf # flag indicating whether or not to weigh the final matrix by term frequency-inverse document frequency (TF-IDF)
        self.ambiguous_rdg_prefix = ambiguous_rdg_prefix # prefix used for ambiguous reading numbers (e.g., "W" or "zw-")
        self.subwitness_suffixes = subwitness_suffixes # list of suffixes used to distinguish subwitnesses like first hands, correctors, main texts, alternate texts, and multiple attestations from their base witnesses
        self.trivial_reading_types = set(trivial_reading_types) # set of reading types (e.g., defective, orthographic, nomSac) whose readings should be collapsed under the previous substantive reading
        self.ignored_reading_types = set(ignored_reading_types) # set of reading types (e.g., lacunose, ambiguous) whose readings should not be included in the matrix
        self.verbose = verbose # flag indicating whether or not to print timing and debugging details for the user
        self.readings = [] # a list of variant reading labels (i.e., the row labels of the matrix)
        self.witnesses = [] # a list of witness sigla (i.e., the column labels of the matrix)
        self.collation_matrix = np.zeros(shape=(len(self.readings), len(self.witnesses)))
        self.fragmentary_witnesses = [] # a list of witnesses with fewer than min_extant extant readings
        self.fragmentary_collation_matrix = np.zeros(shape=(len(self.readings), len(self.fragmentary_witnesses)))

    """
    Postprocesses the collation matrix, moving columns whose coefficients sum below the min_extant threshold to the fragmentary witnesses collation matrix
    and optionally reweighting both matrices by TF-IDF.
    """
    def postprocess(self):
        if self.verbose:
            print("Filtering for witnesses with at least %d extant passages..." % self.min_extant)
        t0 = time.time()
        # Calculate the column sums of the initial collation matrix:
        col_sums = np.sum(self.collation_matrix, axis=0)
        # Then reduce the witnesses list and collation matrix to include just the witnesses that this threshold:
        sufficient_col_inds = [j for j in range(len(self.witnesses)) if col_sums[j] >= self.min_extant]
        fragmentary_col_inds = [j for j in range(len(self.witnesses)) if col_sums[j] < self.min_extant]
        sufficient_witnesses = [self.witnesses[j] for j in sufficient_col_inds]
        fragmentary_witnesses = [self.witnesses[j] for j in fragmentary_col_inds]
        self.witnesses = sufficient_witnesses
        self.fragmentary_witnesses = fragmentary_witnesses
        self.fragmentary_collation_matrix = self.collation_matrix[:, fragmentary_col_inds]
        self.collation_matrix = self.collation_matrix[:, sufficient_col_inds]
        # Now ensure that any readings that are no longer attested among the non-fragmentary witnesses have their rows removed from both matrices:
        row_sums = np.sum(self.collation_matrix, axis=1)
        preserved_row_inds = [i for i in range(len(self.readings)) if row_sums[i] > 0]
        preserved_readings = [self.readings[i] for i in preserved_row_inds]
        self.readings = preserved_readings
        self.collation_matrix = self.collation_matrix[preserved_row_inds, :]
        self.fragmentary_collation_matrix = self.fragmentary_collation_matrix[preserved_row_inds, :]
        t1 = time.time()
        if self.verbose:
            print("Done in %0.4fs." % (t1 - t0))
        # If the use_tfidf flag is set, then reweight the matrices:
        if self.use_tfidf:
            if self.verbose:
                print("Applying TF-IDF reweighting...")
            t0 = time.time()
            tfidf_trans = TfidfTransformer(norm=None, smooth_idf=False) # smooth_idf is not needed, since we've removed all rows that sum to 0, thereby avoiding division by 0
            if self.collation_matrix.shape != (0,0):
                tfidf_trans.fit(self.collation_matrix.T) # fit the transformer only to the non-fragmentary data; we transpose our collation matrix, as scikitlearn expects terms to be in columns and documents in rows
                tfidf_trans.idf_ = tfidf_trans.idf_ - 1 # the TF-IDF transformer adds 1 to each entry; subtract it back out (common readings should be reweighted close to 0, not 1)
                self.collation_matrix = tfidf_trans.transform(self.collation_matrix.T).todense().T
                if self.fragmentary_collation_matrix.shape != (0,0):
                    self.fragmentary_collation_matrix = tfidf_trans.transform(self.fragmentary_collation_matrix.T).todense().T
            # For each fragmentary witnesses, we weigh it readings as if it were the only witness added to the 
            t1 = time.time()
            if self.verbose:
                print("Done in %0.4fs." % (t1 - t0))
        return

"""
Derived class for reading collation data from a TEI XML file.
"""
class tei_collation_parser(collation_parser):
    """
    Given a witness siglum, returns the base siglum of the witness, stripped of all subwitness suffixes.
    """
    def get_base_wit(self, wit):
        base_wit = wit.strip("#") # if the witness siglum is a pointer to an xml:id attribute, remove the "#" prefix
        while (True):
            suffix_found = False
            for suffix in self.subwitness_suffixes:
                if base_wit.endswith(suffix):
                    suffix_found = True
                    base_wit = base_wit[:-len(suffix)]
                    break
            if not suffix_found:
                break
        return base_wit

    """
    Given the XML tree for an element, recursively serializes it in a more readable format.
    """
    def serialize(self, xml):
        # Get the element tag:
        raw_tag = xml.tag.replace("{%s}" % tei_ns, "")
        # If it is a reading, then serialize its children, separated by spaces:
        if raw_tag == "rdg":
            text = "" if xml.text is None else xml.text
            text += " ".join([self.serialize(child) for child in xml])
            return text
        # If it is a word, abbreviation, or overline-rendered element, then serialize its text and tail, 
        # recursively processing any subelements:
        if raw_tag in ["w", "abbr", "hi"]:
            text = "" if xml.text is None else xml.text
            text += "".join([self.serialize(child) for child in xml])
            text += "" if xml.tail is None else xml.tail
            return text
        # If it is a space, then serialize as a single space:
        if raw_tag == "space":
            text = "["
            text += "space"
            if xml.get("reason") is not None:
                text += " "
                reason = xml.get("reason")
                text += "(" + reason + ")"
            if xml.get("unit") is not None and xml.get("extent") is not None:
                text += ", "
                unit = xml.get("unit")
                extent = xml.get("extent")
                text += extent + " " + unit
            text += "]"
            text += "" if xml.tail is None else xml.tail
            return text
        # If it is an expansion, then serialize it in parentheses:
        if raw_tag == "ex":
            text = ""
            text += "("
            text += "" if xml.text is None else xml.text
            text += " ".join([self.serialize(child) for child in xml])
            text += ")"
            text += "" if xml.tail is None else xml.tail
            return text
        # If it is a gap, then serialize it based on its attributes:
        if raw_tag == "gap":
            text = ""
            text += "["
            text += "gap"
            if xml.get("reason") is not None:
                text += " "
                reason = xml.get("reason")
                text += "(" + reason + ")"
                text += reason
            if xml.get("unit") is not None and xml.get("extent") is not None:
                text += ", "
                unit = xml.get("unit")
                extent = xml.get("extent")
                text += extent + " " + unit
            text += "]"
            text += "" if xml.tail is None else xml.tail
            return text
        # If it is an unclear or supplied element, then recursively set the contents in brackets:
        if raw_tag in ["unclear", "supplied"]:
            text = ""
            text += "["
            text += "" if xml.text is None else xml.text
            text += " ".join([self.serialize(child) for child in xml])
            text += "]"
            text += "" if xml.tail is None else xml.tail
            return text
        # If it is a choice element, then recursively set the contents in brackets, separated by slashes:
        if raw_tag == "choice":
            text = ""
            text += "["
            text += "" if xml.text is None else xml.text
            text += "/".join([self.serialize(child) for child in xml])
            text += "]"
            text += "" if xml.tail is None else xml.tail
            return text
        # If it is a ref element, then set its text in brackets:
        if raw_tag == "ref":
            text = ""
            text += "["
            text += "" if xml.text is None else xml.text
            text += "]"
            text += "" if xml.tail is None else xml.tail
            return text
        # For all other elements, return an empty string:
        return ""
        
    """
    Given a TEI XML <app/> element, parses the support for its readings, accounting for subwitness prefixes and reading types.
    The output is a dictionary mapping each base witness to a dictionary of its raw coefficient(s) for the reading(s) it (or any of its subwitnesses) supports.
    """
    def parse_app(self, xml):
        readings_by_witness = {} # the output dictionary
        readings_by_number = {} # dictionary mapping reading numbers to their full labels
        # Get the ID or number of the variation unit:
        app_id = ""
        if xml.get("{%s}id" % xml_ns) is not None:
            app_id = xml.get("{%s}id" % xml_ns)
        elif xml.get("n") is not None:
            app_id = xml.get("n")
        if self.verbose:
            print("Parsing variation unit %s..." % app_id)
        t0 = time.time()
        # In a first loop, populate the readings_by_number dictionary for substantive readings, and add reading labels to the readings list:
        for rdg in xml.xpath("tei:rdg", namespaces={"tei": tei_ns}):
            rdg_type = ""
            if rdg.get("type") is not None:
                rdg_type = rdg.get("type")
            # If this reading is of an ignored or trivial type, then skip it:
            if rdg_type in self.ignored_reading_types or rdg_type in self.trivial_reading_types:
                continue
            rdg_n = rdg.get("n")
            rdg_text = self.serialize(rdg)
            rdg_label = " ".join([app_id, rdg_n, rdg_text])
            # Ambiguous readings should not be added to the readings list, 
            # and their coefficients dictionaries will likely split the unit of support between multiple possible substantive readings:
            if rdg_type != "ambiguous":
                readings_by_number[rdg_n] = rdg_label
                self.readings.append(rdg_label)
        # In a second pass, update the reading coefficients dictionary for each substantive readings and add its value(s) to the reading support dictionaries of all supporting witnesses:
        rdg_coefficients = {}
        for rdg in xml.xpath("tei:rdg", namespaces={"tei": tei_ns}):
            rdg_type = ""
            if rdg.get("type") is not None:
                rdg_type = rdg.get("type")
            # If this reading is of an ignored type, then skip it:
            if rdg_type in self.ignored_reading_types:
                continue
            # If this reading is not of a trivial type, then update the coefficients dictionary:
            if rdg_type not in self.trivial_reading_types:
                rdg_n = rdg.get("n")
                rdg_text = self.serialize(rdg)
                rdg_label = " ".join([app_id, rdg_n, rdg_text])
                rdg_coefficients = {}
                # Ambiguous readings should not be added to the readings list, 
                # and their coefficients dictionaries will likely split the unit of support between multiple possible substantive readings:
                if rdg_type == "ambiguous":
                    # Following the ECM convention, possible readings are separated by forward slashes:
                    possible_reading_numbers = rdg_n.strip(self.ambiguous_rdg_prefix).split("/")
                    # Determine how many of these correspond to substantive readings:
                    for possible_reading_number in possible_reading_numbers:
                        if possible_reading_number in readings_by_number:
                            rdg_coefficients[readings_by_number[possible_reading_number]] = 1
                    # If the ambiguous reading corresponds to no substantive reading, then ignore it;
                    # otherwise, normalize the coefficients to sum to 1:
                    if len(rdg_coefficients) == 0:
                        continue
                    else:
                        for label in rdg_coefficients:
                            rdg_coefficients[label] = 1 / len(rdg_coefficients)
                else:
                    rdg_coefficients[rdg_label] = 1
            # Now add the coefficients for this reading to the support dictionary of every witness supporting this reading:
            if rdg.get("wit") is None:
                continue
            for wit in rdg.get("wit").split():
                # Extract the base siglum for this witness:
                base_wit = self.get_base_wit(wit)
                # If the readings_by_witness dictionary doesn't have an entry for this witness yet, then create one:
                if base_wit not in readings_by_witness:
                    readings_by_witness[base_wit] = {}
                # Then add the coefficient(s) for this reading to this witness's dictionary:
                base_wit_coefficients = readings_by_witness[base_wit]
                for label in rdg_coefficients:
                    if label not in base_wit_coefficients:
                        base_wit_coefficients[label] = 0
                    base_wit_coefficients[label] += rdg_coefficients[label]
                readings_by_witness[base_wit] = base_wit_coefficients
        t1 = time.time()
        if self.verbose:
            print("Done in %0.4fs." % (t1 - t0))
        return readings_by_witness

    """
    Given a file address to an .xml file containing a TEI collation, read its contents into a collation matrix and apply the appropriate postprocessing.
    """
    def read(self, input_addr):
        # Parse the input XML document:
        xml = et.parse(input_addr)
        # Reinitialize the reading and witness lists:
        self.readings = []
        self.witnesses = []
        # Populate them and a dictionary mapping witnesses to a dictionary of their reading coefficients, parsing one variation unit at a time:
        if self.verbose:
            print("Parsing variation units in TEI XML tree...")
        t0 = time.time()
        readings_by_witness = {}
        # Set the minimum extant readings threshold based on the number of variation units in the input:
        self.min_extant = int(self.min_extant_proportion * len(xml.xpath("//tei:app", namespaces={"tei": tei_ns})))
        for app in xml.xpath("//tei:app", namespaces={"tei": tei_ns}):
            readings_by_witness_for_app = self.parse_app(app)
            # Proceed for every (base) witness accounted for in this unit:
            for wit in readings_by_witness_for_app:
                # If this witness has not yet appeared, then add it to the list and initialize its reading coefficient dictionary:
                if wit not in readings_by_witness:
                    self.witnesses.append(wit)
                    readings_by_witness[wit] = {}
                # Then add its coefficients for this unit to its reading coefficient dictionary:
                wit_coefficients = readings_by_witness[wit]
                wit_coefficients_for_app = readings_by_witness_for_app[wit]
                for label in wit_coefficients_for_app:
                    wit_coefficients[label] = wit_coefficients_for_app[label]
                readings_by_witness[wit] = wit_coefficients
        # Now populate the collation matrix, using dictionaries to map values to row and column indices:
        rows_by_reading = {} # a dictionary mapping variant reading labels to their corresponding row indices
        cols_by_witness = {} # a dictionary mapping base witness sigla to their corresponding column indices
        for i, rdg in enumerate(self.readings):
            rows_by_reading[rdg] = i
        for j, wit in enumerate(self.witnesses):
            cols_by_witness[wit] = j
        self.collation_matrix = np.zeros(shape=(len(self.readings), len(self.witnesses)))
        for wit in readings_by_witness:
            j = cols_by_witness[wit]
            wit_coefficients = readings_by_witness[wit]
            for rdg in wit_coefficients:
                i = rows_by_reading[rdg]
                coefficient = wit_coefficients[rdg]
                self.collation_matrix[i,j] = coefficient
        t1 = time.time()
        if self.verbose:
            print("Total time to parse all %d variation units: %0.4fs." % (len(xml.xpath("//tei:app", namespaces={"tei": tei_ns})), t1 - t0))
        if self.verbose:
            print("Size of raw collation matrix: %d rows (readings), %d columns (witnesses)." % self.collation_matrix.shape)
        # Finally, postprocess this matrix:
        self.postprocess()
        if self.verbose:
            print("Size of primary collation matrix: %d rows (readings), %d columns (witnesses)." % self.collation_matrix.shape)
            print("Size of fragmentary witnesses collation matrix: %d rows (readings), %d columns (witnesses)." % self.fragmentary_collation_matrix.shape)
        return

"""
Derived class for reading collation data from the INTF Virtual Manuscript Room (VMR).
"""
class vmr_collation_parser(collation_parser):
    manuscript_witness_pattern = re.compile(r"^[PL]*\d+")
    fehler_pattern = re.compile(r"f\d*$")
    defective_rdg_pattern = re.compile(r"^[a-z]+f\d*$")
    orthographic_rdg_pattern = re.compile(r"^[a-z]+o\d*$")
    overlap_rdg_label = "zu"
    ambiguous_rdg_label = "zw"
    lacunose_rdg_label = "zz"

    """
    Given a string of witness sigla, remove any square brackets around witness sigla.
    """
    def remove_square_brackets(self, wit_str):
        debracketed_wit_str = wit_str.replace("[", "").replace("]", "")
        return debracketed_wit_str

    """
    Given a string of witness sigla, remove any right angle brackets after versional witness sigla.
    """
    def remove_angle_brackets(self, wit_str):
        debracketed_wit_str = wit_str.replace(">", "")
        return debracketed_wit_str

    """
    Given a string of witness sigla, expand any base sigla followed by one or more suffixes in the same parentheses to the same sigla followed by each suffix
    """
    def expand_parenthetical_suffixes(self, wit_str):
        expanded_wit_str = wit_str
        witness_with_parentheses_pattern = re.compile(r"(\S+)\(([^\(\)]*)\)")
        matches = witness_with_parentheses_pattern.findall(wit_str)
        for match in matches:
            wit = self.get_base_wit(match[0])
            suffixes = match[1].replace(" ", "").split(",")
            expanded_wits = []
            for suffix in suffixes:
                expanded_wit = wit + suffix
                expanded_wits.append(expanded_wit)
            expanded_wit_str = expanded_wit_str.replace(match[0] + "(" + match[1] + ")", " ".join(expanded_wits))
        return expanded_wit_str

    """
    Given a string of witness sigla, remove any "ms" and "mss" suffixes after patristic and versional witness sigla.
    """
    def remove_ms_mss_suffixes(self, wit_str):
        reduced_sigla = wit_str.split()
        for siglum in reduced_sigla:
            # Only check this for non-manuscript witnesses:
            if not self.manuscript_witness_pattern.match(siglum):
                if siglum.endswith("mss"):
                    siglum = siglum[:-3]
                elif siglum.endswith("ms"):
                    siglum = siglum[:-2]
        reduced_wit_str = " ".join(reduced_sigla)
        return reduced_wit_str

    """
    Given a witness siglum, returns the base siglum of the witness, stripped of all subwitness suffixes.
    """
    def get_base_wit(self, wit):
        base_wit = wit
        # Strip any et videtur and defective suffixes and any user-specified subwitness suffixes:
        while (True):
            suffix_found = False
            for suffix in self.subwitness_suffixes:
                if base_wit.endswith(suffix):
                    suffix_found = True
                    base_wit = base_wit[:-len(suffix)]
            if suffix_found:
                continue
            if self.fehler_pattern.search(base_wit):
                suffix_found = True
                fehler_suffix = self.fehler_pattern.search(base_wit).group()
                base_wit = base_wit[:-len(fehler_suffix)]
                continue
            if base_wit.endswith("V"):
                suffix_found = True
                base_wit = base_wit[:-1]
                continue
            if not suffix_found:
                break
        return base_wit

    """
    Given an XML <segment/> element, parses the support for its readings, accounting for subwitness prefixes and reading types.
    The output is a dictionary mapping each base witness to a dictionary of its raw coefficient(s) for the reading(s) it (or any of its subwitnesses) supports.
    """
    def parse_segment(self, xml):
        readings_by_witness = {} # the output dictionary
        readings_by_number = {} # dictionary mapping reading numbers to their full labels
        # Get the full location string for the variation unit:
        segment_id = xml.get("verse") + "/" + xml.get("wordsegs")
        if self.verbose:
            print("Parsing variation unit %s..." % segment_id)
        t0 = time.time()
        # In a first loop, populate the readings_by_number dictionary for substantive readings, and add reading labels to the readings list:
        for rdg in xml.xpath(".//segmentReading"):
            # Determine the type of this reading from its label:
            rdg_type = ""
            rdg_label = rdg.get("label").replace("♦", "").strip() # remove diamonds and surrounding whitespace
            if rdg_label == self.lacunose_rdg_label:
                rdg_type = "lac"
            if rdg_label == self.ambiguous_rdg_label:
                rdg_type = "ambiguous"
            if rdg_label == self.overlap_rdg_label:
                rdg_type = "overlap"
            if self.orthographic_rdg_pattern.match(rdg_label):
                rdg_type = "orthographic"
            if self.defective_rdg_pattern.match(rdg_label):
                rdg_type = "defective"
            # If this reading is of an ignored or trivial type, then skip it:
            if rdg_type in self.ignored_reading_types or rdg_type in self.trivial_reading_types:
                continue
            rdg_text = rdg.get("reading")
            rdg_id = " ".join([segment_id, rdg_label, rdg_text])
            # Ambiguous readings should not be added to the readings list, 
            # and their coefficients dictionaries will likely split the unit of support between multiple possible substantive readings:
            if rdg_type != "ambiguous":
                readings_by_number[rdg_label] = rdg_id
                self.readings.append(rdg_id)
        # In a second pass, update the reading coefficients dictionary for each substantive readings and add its value(s) to the reading support dictionaries of all supporting witnesses:
        rdg_coefficients = {}
        for rdg in xml.xpath(".//segmentReading"):
            # Determine the type of this reading from its label:
            rdg_type = ""
            rdg_label = rdg.get("label").replace("♦", "").strip() # remove diamonds and surrounding whitespace
            if rdg_label == self.lacunose_rdg_label:
                rdg_type = "lac"
            if rdg_label == self.ambiguous_rdg_label:
                rdg_type = "ambiguous"
            if rdg_label == self.overlap_rdg_label:
                rdg_type = "overlap"
            if self.orthographic_rdg_pattern.match(rdg_label):
                rdg_type = "orthographic"
            if self.defective_rdg_pattern.match(rdg_label):
                rdg_type = "defective"
            # If this reading is of an ignored type, then skip it:
            if rdg_type in self.ignored_reading_types:
                continue
            # If this reading is not of a trivial type, then update the coefficients dictionary:
            if rdg_type not in self.trivial_reading_types:
                rdg_label = rdg.get("label").replace("♦", "").strip() # remove diamonds and surrounding whitespace
                rdg_text = rdg.get("reading")
                rdg_id = " ".join([segment_id, rdg_label, rdg_text])
                rdg_coefficients = {}
                # Ambiguous readings should not be added to the readings list, 
                # and their coefficients dictionaries will likely split the unit of support between multiple possible substantive readings:
                if rdg_type == "ambiguous":
                    # Following the ECM convention, possible readings are separated by forward slashes:
                    possible_reading_labels = rdg_text.replace("_f", "").split("/") # remove the defective suffix
                    # Determine how many of these correspond to substantive readings:
                    for possible_reading_label in possible_reading_labels:
                        if possible_reading_label in readings_by_number:
                            rdg_coefficients[readings_by_number[possible_reading_label]] = 1
                    # If the ambiguous reading corresponds to no substantive reading, then ignore it;
                    # otherwise, normalize the coefficients to sum to 1:
                    if len(rdg_coefficients) == 0:
                        continue
                    else:
                        for rdg_coefficients_id in rdg_coefficients:
                            rdg_coefficients[rdg_coefficients_id] = 1 / len(rdg_coefficients)
                else:
                    rdg_coefficients[rdg_id] = 1
            # Now preprocess the witnesses string and add the coefficients for this reading to the support dictionary of every witness supporting this reading:
            if rdg.get("witnesses") is None:
                continue
            processed_witnesses_str = rdg.get("witnesses")
            processed_witnesses_str = self.remove_square_brackets(processed_witnesses_str)
            processed_witnesses_str = self.remove_angle_brackets(processed_witnesses_str)
            processed_witnesses_str = self.expand_parenthetical_suffixes(processed_witnesses_str)
            processed_witnesses_str = self.remove_ms_mss_suffixes(processed_witnesses_str)
            for wit in processed_witnesses_str.split():
                # Is this a manuscript witness?
                if self.manuscript_witness_pattern.match(wit):
                    # Extract the base siglum for this witness:
                    base_wit = self.get_base_wit(wit)
                    # If the readings_by_witness dictionary doesn't have an entry for this witness yet, then create one:
                    if base_wit not in readings_by_witness:
                        readings_by_witness[base_wit] = {}
                    # Then add the coefficient(s) for this reading to this witness's dictionary:
                    base_wit_coefficients = readings_by_witness[base_wit]
                    for label in rdg_coefficients:
                        if label not in base_wit_coefficients:
                            base_wit_coefficients[label] = 0
                        base_wit_coefficients[label] += rdg_coefficients[label]
                    readings_by_witness[base_wit] = base_wit_coefficients
                # TODO: Fathers and versions are tricky to parse unambiguously; it would be ideal to retrieve and parse them separately?
                # For now, we'll just stop after we've processed the manuscript witnesses...
                else:
                    break
        t1 = time.time()
        if self.verbose:
            print("Done in %0.4fs." % (t1 - t0))
        return readings_by_witness

    """
    Given a content tag (e.g., "Acts.1.1-5" for Acts 1:1-5, "Acts.5" for Acts 5, or "Acts" for all of Acts) to an .xml file containing a TEI collation, read its contents into a collation matrix and apply the appropriate postprocessing.
    """
    def read(self, index):
        # Retrieve the collation data for the desired indices:
        if self.verbose:
            print("Requesting ECM collation data for %s..." % index)
        t0 = time.time()
        # Get the appropriate project name based on the book in question:
        request_str = "https://ntvmr.uni-muenster.de/community/vmr/api/variant/apparatus/get/?indexContent=%s&positiveConversion=true&buildA=false&format=xml" % index
        # Parse the contents of the HTTP request as an XML string:
        xml = None
        with urllib.request.urlopen(request_str) as r:
            contents = r.read()
            xml = et.fromstring(contents)
        t1 = time.time()
        if self.verbose:
            print("Done in %0.4fs." % (t1 - t0))
        # Reinitialize the reading and witness lists:
        self.readings = []
        self.witnesses = []
        # Populate them and a dictionary mapping witnesses to a dictionary of their reading coefficients, parsing one variation unit at a time:
        if self.verbose:
            print("Parsing variation units in XML response...")
        t0 = time.time()
        readings_by_witness = {}
        # Set the minimum extant readings threshold based on the number of variation units in the input:
        self.min_extant = int(self.min_extant_proportion * len(xml.xpath("//segment")))
        for segment in xml.xpath("//segment"):
            readings_by_witness_for_segment = self.parse_segment(segment)
            # Proceed for every (base) witness accounted for in this unit:
            for wit in readings_by_witness_for_segment:
                # If this witness has not yet appeared, then add it to the list and initialize its reading coefficient dictionary:
                if wit not in readings_by_witness:
                    self.witnesses.append(wit)
                    readings_by_witness[wit] = {}
                # Then add its coefficients for this unit to its reading coefficient dictionary:
                wit_coefficients = readings_by_witness[wit]
                wit_coefficients_for_segment = readings_by_witness_for_segment[wit]
                for label in wit_coefficients_for_segment:
                    wit_coefficients[label] = wit_coefficients_for_segment[label]
                readings_by_witness[wit] = wit_coefficients
        # TODO: If we can retrieve fathers and versions separately, we would request and process their variation units in separate loops next.
        # Now populate the collation matrix, using dictionaries to map values to row and column indices:
        rows_by_reading = {} # a dictionary mapping variant reading labels to their corresponding row indices
        cols_by_witness = {} # a dictionary mapping base witness sigla to their corresponding column indices
        for i, rdg in enumerate(self.readings):
            rows_by_reading[rdg] = i
        for j, wit in enumerate(self.witnesses):
            cols_by_witness[wit] = j
        self.collation_matrix = np.zeros(shape=(len(self.readings), len(self.witnesses)))
        for wit in readings_by_witness:
            j = cols_by_witness[wit]
            wit_coefficients = readings_by_witness[wit]
            for rdg in wit_coefficients:
                i = rows_by_reading[rdg]
                coefficient = wit_coefficients[rdg]
                self.collation_matrix[i,j] = coefficient
        t1 = time.time()
        if self.verbose:
            print("Total time to parse all %d variation units: %0.4fs." % (len(xml.xpath("//segment")), t1 - t0))
        if self.verbose:
            print("Size of raw collation matrix: %d rows (readings), %d columns (witnesses)." % self.collation_matrix.shape)
        # Finally, postprocess this matrix:
        self.postprocess()
        if self.verbose:
            print("Size of primary collation matrix: %d rows (readings), %d columns (witnesses)." % self.collation_matrix.shape)
            print("Size of fragmentary witnesses collation matrix: %d rows (readings), %d columns (witnesses)." % self.fragmentary_collation_matrix.shape)
        return