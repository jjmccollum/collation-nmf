# collation-nmf

General-purpose Python tools to perform non-negative matrix factorization on textual collations

## About This Project

Initially, I had a specific need for a utility that could parse a large (> 10000 lines) collation in TEI XML format and convert it to a collation matrix that could be subjected to machine learning techniques.
I later decided to extend this utility into a more general-purpose tool for analyzing textual collations on the user's machine (in TEI XML format) or from the INTF's New Testament Virtual Manuscript Room (NTVMR).
The conversion to matrix format is handled in the `collation_parser` base class and its extensions, which may be of independent value for other projects.

To extract underlying groups' reading profiles and witness memberships together, this tool applies non-negative matrix factorization (NMF) to the collation matrix.
This approach has been applied to Tommy Wasserman's extensive collation of the epistle of Jude with promising results (Joey McCollum, "Biclustering Readings and Manuscripts via Non-negative Matrix Factorization, with Application to the Text of Jude," _Andrews University Seminary Studies_ 57.1 \[2019\], 61–89, https://digitalcommons.andrews.edu/auss/vol57/iss1/6/).
Because it describes data points (i.e., witnesses) according to an additive mixture model, NMF is able to accommodate contamination in a textual tradition (although it should be noted that because NMF makes no judgments on the genealogical priority of readings, later readings can be weighed as more significant in a group's profile, and isolated early witnesses that share early readings found in multiple groups may appear to have mixture from these groups).
This model also allows NMF to accommodate "polysemy" of readings, or coincidental agreement between groups.
This tool uses the `nimfa` Python library (Marinka Žitnik and Blaž Zupan, "NIMFA: A Python Library for Nonnegative Matrix Factorization," _Journal of Machine Learning Research_ 13 \[2012\], 849–853, DOI:[10.5555/2503308.2188415](https://dl.acm.org/doi/10.5555/2503308.2188415); software documentation at https://nimfa.biolab.si/) for NMF.

To facilitate the identification of better-separated witness groups with more exclusive readings, the collation matrix can be reweighted according to the term frequency-inverse document frequency (TF-IDF) scheme before NMF is applied.
This reweighting is handled by the `collation_parser` class, and it uses the `TfIdfTransformer` class from the `scikit-learn` machine learning library (https://scikit-learn.org/).
The factorization itself and rank estimation (i.e., evaluating how well the data fits a given number of groups) are handled by the `collation_factorizer` class.
Rank estimation for a range of target numbers of groups can be done with the `py/estimate_rank.py` script.
Because manuscript data is naturally generated according to a hierarchical process (descent with modification), most ranks will likely produce good results (with the main difference being the granularity of distinctions between groups), so rank estimation may not be necessary.
Factorization for a single number of groups is handled by the `py/factorize_collation.py` script.
The resulting basis (i.e., reading profile) and mixture (i.e., witness grouping) factor matrices can be written to Excel (`.xlsx`) or JSON (`.json`) output.

Because highly fragmentary witnesses can hinder the optimization process behind NMF, it is best to set aside witnesses that are not sufficiently complete for the "training" step of the factorization.
After NMF has extracted factors based on more complete data, these set-aside witnesses can then be classified using the extracted group reading profiles.
Computationally, this is done by solving a non-negative least squares (NNLS) optimization problem involving each set-aside witness and the basis matrix.
This is also handled by the `py/factorize_collation.py` script, and the classifications of the set-aside witnesses are included in its output.
The NNLS problems are solved using the SciPy library (https://scipy.org/).

## Getting Started

It is assumed that you have Python 3 (this code has not been tested with Python 2 and probably will not work with it!) installed on your system.
If you don't, you can get it for free at https://www.python.org/.

From this page, you can either download a `.zip` archive of the project to the desired location on your system, or if you have git installed (https://git-scm.com/), you can clone this repository to the desired location using the instructions in the "Code" tab.
Then, to install all necessary dependencies, enter the following command from the project directory:

```
pip install py/
```

If using Python 3 on Linux, you may have to use `pip3` instead of `pip` above and `python3` instead of `python` to run the scripts.
If you are using Windows, then you should use a backslash (`\`) instead of a forward slash (`/`) for commands involving directories.

## Usage

### Inputs and Preprocessing

The `collation_parser` base class is extended into two derived classes that handle TEI XML collations and NTVMR data, respectively.
Reading TEI XML input is faster (because the HTTP requests needed for querying the NTVMR are not needed), but if you want to use your own TEI XML collation data, you will want to keep some rules in mind.

In an unabridged collation, substantive variant readings will sometimes have their own subvariants (e.g., defective spellings, orthographic alternatives, or, in the case of the CBGM, split attestations), while other types of readings (e.g., placeholder readings indicating overlap from a separate variation unit, ambiguous transcriptions that could resolve to multiple substantive readings, or lacunae) may be listed in addition to substantive readings for completeness.
In TEI XML inputs, the nature of a reading is expected to be listed under its `type` attribute.
This is important, because the `collation_parser` class will search for this attribute in each reading, and based on user-specified parameters, it will do one of the following things:

1. ignore the reading and do not add entries for its support to the collation matrix (typically how lacunae are handled, and the same may be done for ambiguous readings);
2. treat the reading as trivial, adding its support to the last substantive reading encountered (typically how some or all types of subvariants are handled); or
3. treat the reading as substantive, adding its support to entries in the matrix.

Which types of readings are handled in which ways can be specified by the user: the `-z` flag will indicate which readings to ignore, and the `-t` flag will indicate which readings to treat as trivial.
All readings of other types will be treated as substantive.

Readings of type `ambiguous`, if they are not ignored, are handled slightly differently.
They are not treated as substantive readings in their own right; rather, the contribution of their support is split over all the substantive readings they could potentially be, which are indicated by an ambiguous reading prefix (typically something like "W" or "zw-"; this can be specified by the user using the `-w` flag) and the possible options separated by forward slashes.
So, an ambiguous reading with label "zw-a/b" would contribute an entry of 1/2 to reading a and 1/2 to reading b.

Due to factors detailed above, it is recommended that you structure readings in your own TEI XML collation file accordingly. Consider the following encoding of a variation unit:

```xml
<app n="1">
    <rdg n="1" wit="A">...</rdg>
    <rdg n="1-f1" type="defective" wit="B">...</rdg>
    <rdg n="1-f2" type="defective" wit="C">...</rdg>
    <rdg n="1-o1" type="orthographic" wit="D">...</rdg>
    <rdg n="1-o1-f1" type="defective" wit="E">...</rdg>
    <rdg n="2" wit="F">...</rdg>
    <rdg n="Z" type="lac" wit="G">...</rdg>
</app>
```

If we parse this collation using the flags

```
-t defective -z lac
```

then witnesses `A`, `B`, and `C` will all have entries for reading `1` (because the defective readings `1-f1` and `1-f2` are collapsed into the last substantive reading `1`), witnesses `D` and `E` will both have entries for reading `1-o1` (because orthographic readings are not specified as trivial, and the defective reading `1-o1-f1` is collapsed into this reading), witness `F` will have an entry for reading `2`, and witness `G` will not be listed under any reading (because reading `Z`, which encodes a lacuna, is ignored).

Witness sigla are assumed to follow ECM numbering conventions for manuscripts (i.e., a number, optionally preceded by "P" or "L"), but suffixes can be handled accordingly.
Suffixes that should be treated as trivial can be specified with the `-s` command.
For example, if first-hand and main-text sigla "424*" and "424T" should be identified with their base witness siglum "424", then the options `-s "*" -s "T"` should be specified on the command line.
In this case, correctors and marginal/alternate/commentary/etc. readings associated with the same base witness will be treated as distinct (probably fragmentary) witnesses.
Often, lectionaries or other manuscripts may repeat the same passage multiple times, and differences between the multiple copies of the text may be distinguished using sigla like "/1" and "/2" or "L1" and "L2".
If all of these suffixes are specified as trivial (via `-s "/1" -s "/2"`), then the readings from all repeated passages will be added together under the siglum of the base witness.
In the additive mixture model of NMF, the influence of different sources on the multiple versions of the passage should still be detectable if it is strong enough.

In any of the scripts included in the projects, if you want to see printouts of status updates and operation timing (for the purposes of debugging or tracking performance), then simply specify the `--verbose` command.

### Collation Matrix Postprocessing

By default, the `collation_parser` class only includes manuscripts whose reading count is at least 95% of the number of variation units in the collation, setting aside all other witnesses as fragmentary.
This can be changed by specifying a different proportion between 0 and 1 with the `-p` parameter (e.g., `-p 0.9` to include manuscripts whose reading count is at least 90% of the number of variation units).
Note that since readings found only in set-aside witnesses are not used in NMF, these readings will be discarded from the analysis.

By default, the collation matrix is initialized using simple counts of readings for witnesses.
To use TF-IDF weighting (which is recommended), include the `--use-tfidf` flag.

### Rank Estimation

Rank estimation is done using the least-squares NMF (`Lsnfm`) variant of NMF implemented in `nimfa`.
The factor matrices are initialized using the `random_vcol` initialization rule, and they are improved over 10 iterations for each trial.
The number of NMF trials to run is 10 by default, but can be specified using the `-nrun` parameter.
More trials will yield more reliable results at the expense of a longer running time.

The output of rank estimation is a set of the following metrics for each rank (i.e., target number of groups):
- `cophenetic`: the cophenetic correlation coefficient. Essentially, a measurement of how consistently groups are formed when NMF is run with different starting guesses. Values closer to 1 are better, and values closer to 0 are worse. A good rule of thumb is to use the rank at which this value begins to drop.
- `rss`: the residual sum of squares of the difference between the collation matrix and the product of the factor matrices. In simpler terms, NMF tries to reconstruct the original collation matrix as a product of two simple factors, and the residual sum of squares measures how "off" this reconstruction is. Thus, `rss` values closer to 0 are better.
- `evar`: the proportion of explained variance between the collation matrix and the product of the factor matrices. Values closer to 1 are better, while values closer to 0 are worse.
- `basis_sparseness`: the (column) sparseness of the basis (i.e., reading profile) and mixture (i.e., witness grouping) factor matrix, measured as a proportion between 0 and 1. If close to 1, this indicates that the group profiles generally consist of fewer, more characteristic readings.
- `mixture_sparseness`: the (column) sparseness of the mixture (i.e., witness grouping) factor matrix, measured as a proportion between 0 and 1. If close to 1, this indicates that the witnesses feature less mixture from multiple profiles.

By default, the table of rank estimation measures is printed to the console (since it is small enough to be easily readable),
but if the user specifies an output Excel (`.xlsx`) or JSON (`.json`) file with the `-o` argument, then the output will be written to that file instead.

The required arguments of the `estimate_rank.py` script are the input (either a `.xml` collation file or a content index for the NTVMR, such as `"Acts.1-10"`), a minimum rank at which to start the search, and a maximum rank at which to stop the search.

### Factorization of Collation Matrix and Classification of Fragmentary Witnesses

The factorization proper is done using the least-squares NMF (`Lsnfm`) variant of NMF implemented in `nimfa`.
The factor matrices are initialized using the `nndsvd` initialization rule, and they are improved over 100 iterations for each trial.

The required arguments of the `factorize_collation.py` script are the input (either a `.xml` collation file or a content index for the NTVMR), the output file (`.xlsx` and `.json` are supported), and rank (i.e., desired number of groups) of the factorization.

## Examples

In the examples that follow, it is assumed that you have navigated your terminal/command prompt to the top directory of this project.

### Rank Estimation, TEI XML Input

To estimate the best rank between 2 and 30 with 100 NMF trials for each rank using TEI XML input (the ECM 3 John collation in the `example` directory), setting aside witnesses with fewer readings than 95% of the number of variation units, and applying TF-IDF reweighting:

```
python py/estimate_rank.py -w "zw-" -s "*" -s "T" -t defective -z lac -p 0.95 --use-tfidf -nrun 100 --verbose -o 3_john_rank_estimation_tfidf_2_30.xlsx example/3_john_collation.xml 2 30
```

Here, defective readings are treated as trivial, and lacunae are ignored.
Ambiguous readings have the prefix `zw-` before their potential disambiguations. 
Witnesses with the first hand or main text suffixes are treated as equivalent to their base witnesses.
Since `--verbose` flag is specified, status messages will be printed to the console over the process.
The output will be written to the Excel file `3_john_rank_estimation_tfidf_2_30.xlsx`.

### Rank Estimation, NTVMR Input

To estimate the best rank between 2 and 20 with 10 NMF trials for each rank using the ECM collation data for Acts chapters 1 through 5, but setting aside witnesses with fewer readings than 90% of the number of variation units:

```
python py/estimate_rank.py -s "*" -s "T" -s "T1" -s "T2" -s "L1" -s "L2" -z lac -p 0.90 --use-tfidf -nrun 10 "Acts.1-5" 2 20
```

Because defective readings are generally already collapsed with their parent substantive readings in the NTVMR data for ECM Acts, the `t defective` argument has been dropped.
The ECM Acts collation uses `T1` and `T2` in addition to `T` as main text sigla, so they have been included as trivial suffixes, as well.
To accommodate lectionaries with multiple versions of the same passage, witnesses with the `L1` and `L2` suffixes (used for this purpose in ECM Acts) are also being treated as equivalent to their base witnesses.
Note that the input content index (`Acts.1-5`) is enclosed in quote marks because otherwise its hyphen would be read as another command-line argument.
Since `--verbose` flag is specified, status messages will be printed to the console over the process.
Since no external output is specified with the `-o` argument, the rank estimation results will be written to the console.

### Factorization, TEI XML Input

To perform a full factorization of the ECM 3 John TEI XML collation data into 14 clusters using the same processing parameters as before:

```
python py/factorize_collation.py -w "zw-" -s "*" -s "T" -t defective -z lac -p 0.95 --use-tfidf example/3_john_collation.xml 3_john_tfidf_rank_14_results.xlsx 14
```

Since `--verbose` flag is not specified, no status messages will be printed to the console.
The output (consisting of a summary sheet, basis and mixture matrices, and a fragmentary witness classification matrix) will be written to the `3_john_tfidf_rank_14_results.xlsx` Excel file.

### Rank Estimation, NTVMR Input

To perform a full factorization of the entire ECM Acts collation into 12 clusters using the same processing parameters as before:

```
python py/factorize_collation.py -s "*" -s "T" -s "T1" -s "T2" -s "L1" -s "L2" -z lac -p 0.90 --use-tfidf Acts acts_tfidf_rank_12_results.xlsx 12
```

Since `--verbose` flag is not specified, no status messages will be printed to the console.
Note that the input content index (`Acts`) does not need to be enclosed in quote marks because it does not contain a hyphen.
The output will be written to the `acts_tfidf_rank_12_results.xlsx` Excel file.

## Potential Improvements

- The current rules for collapsing trivial readings into the last substantive reading assumes a clean hierarchy of potentially trivial reading types (e.g., `orthographic` > `defective`) so that it can process readings in sequence. But this does not work if two types of potentially trivial readings are parallel on this hierarchy: are subvariants involving _plene_ spelling of _nomina sacra_ more substantive than defective subvariants? Less substantive than orthographic variants? A cleaner solution would probably be to have readings point to their parent readings through an attribute. The use of `<rdgGrp/>` elements suggested at https://tei-c.org/release/doc/tei-p5-doc/en/html/TC.html#TCAPSU might also work, but it seems unncessarily bulky.
- In the same way, it would be cleaner if readings of type `ambiguous` pointed to their potential disambiguations through an attribute. Requiring the user to specify how they mark ambiguous readings and then parsing the reading number string is somewhat ad hoc.
- Currently, the only witnesses that are parsed and added to the collation matrix are manuscripts numbered according to the ECM convention. Because the sigla for versions and church fathers do not follow this convention, they are not parsed. The usual suffixes appended to manuscript sigla (e.g., "T", "C", "A") occur in the base sigla of fathers and versions (and may _also_ occur as suffixes to these sigla; consider, for instance, the marginal readings of the Harklean Syriac version, which are printed as "S:HA" in the ECM), so a simple loop checking for and stripping them will not work for these witnesses. A set of sigla similar to those used for papyri and lectionaries (e.g., "F1" for church fathers and "V1" for versions) would work, but there is no precedent for using such a system with non-manuscript witnesses in New Testament textual criticism.

Feel free to add any suggested improvements, feature requests, or bug reports in the issues for this repository.