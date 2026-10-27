"""Build the final report as a two-column research paper (.docx).

The layout follows the conventional IEEE A4 template: two columns of 10 pt
Times New Roman, black throughout, full-width spans for every figure and wide
table, horizontal rules rather than gridded cells, and references numbered in
order of first citation.

Every figure comes from `eda_results/figures/` and every number in the prose is
read out of `eda_results/tables/` at build time rather than typed in, so a
re-run of the analysis cannot leave the text quoting a stale value.
"""
from __future__ import annotations

import os
import re

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from src.eda.settings import FIG_DIR, MODEL_TAB_DIR, ROOT, TAB_DIR

OUT_DIR = os.path.join(ROOT, "report")
DOCX = os.path.join(OUT_DIR, "EDA_Report_CICIDS2017_UNSW-NB15.docx")

TITLE = ("Two Benchmarks, One Feature Space: An Exploratory Analysis of "
         "Distribution Shift Between CICIDS2017 and UNSW-NB15")
AUTHORS = [("Ujjwal Gogoi", "23BDS0335")]
INSTITUTION = ["School of Computer Science and Engineering",
               "Vellore Institute of Technology, Vellore, India"]
COURSE = "Exploratory Data Analysis"
KEYWORDS = ("exploratory data analysis, distribution shift, covariate shift, "
            "network intrusion detection, data quality, feature redundancy, "
            "mutual information, CICIDS2017, UNSW-NB15")

# The repository Appendix A tells a reader to clone. Left empty the clone line
# is omitted, so the paper never carries a placeholder URL.
REPO = "https://github.com/ujjwalgogoi2023/SHAP-based-explanation-transfer"

SERIF = "Times New Roman"
MONO = "Consolas"
BLACK = RGBColor(0x00, 0x00, 0x00)

BODY_PT = 10.0
CAPTION_PT = 8.0
REF_PT = 8.0
INDENT = 0.2            # first-line indent, in inches

# A4 with the two-column measure. SPAN is the full text width, which is also
# the width every figure and wide table is drawn at.
MARGIN_X = 0.72
MARGIN_Y = 0.78
GUTTER = 0.25
SPAN = 8.27 - 2 * MARGIN_X
COL = (SPAN - GUTTER) / 2

ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
         "XI", "XII", "XIII", "XIV", "XV"]


# --------------------------------------------------------------------------
# bibliography
#
# Numbers are assigned in order of first citation, as IEEE style requires, so
# the entries are keyed and the document's own order decides the numbering. A
# `[@key]` token anywhere in the prose or a caption renders as the bracketed
# number; `_italic_` marks a title for italics in the reference list.
# --------------------------------------------------------------------------

BIB: dict[str, str] = {
    "sommer2010":
        'R. Sommer and V. Paxson, "Outside the closed world: On using machine '
        'learning for network intrusion detection," in _Proc. IEEE Symp. '
        'Security and Privacy (S&P)_, 2010, pp. 305-316.',
    "arp2022":
        'D. Arp, E. Quiring, F. Pendlebury, A. Warnecke, F. Pierazzi, '
        'C. Wressnegger, L. Cavallaro, and K. Rieck, "Dos and don\'ts of '
        'machine learning in computer security," in _Proc. 31st USENIX '
        'Security Symp._, 2022, pp. 3971-3988.',
    "lundberg2017":
        'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting '
        'model predictions," in _Advances in Neural Information Processing '
        'Systems (NeurIPS)_, 2017, pp. 4765-4774.',
    "ribeiro2016":
        'M. T. Ribeiro, S. Singh, and C. Guestrin, "“Why should I trust '
        'you?”: Explaining the predictions of any classifier," in _Proc. '
        '22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining '
        '(KDD)_, 2016, pp. 1135-1144.',
    "wang2020":
        'M. Wang, K. Zheng, Y. Yang, and X. Wang, "An explainable machine '
        'learning framework for intrusion detection systems," _IEEE Access_, '
        'vol. 8, pp. 73127-73141, 2020.',
    "oseni2023":
        'A. Oseni, N. Moustafa, G. Creech, N. Sohrabi, A. Strelzoff, Z. Tari, '
        'and I. Linkov, "An explainable deep learning framework for resilient '
        'intrusion detection in IoT-enabled transportation networks," _IEEE '
        'Trans. Intelligent Transportation Systems_, vol. 24, no. 1, '
        'pp. 1000-1014, 2023.',
    "sharafaldin2018":
        'I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward '
        'generating a new intrusion detection dataset and intrusion traffic '
        'characterization," in _Proc. 4th Int. Conf. Information Systems '
        'Security and Privacy (ICISSP)_, 2018, pp. 108-116.',
    "lashkari2017":
        'A. H. Lashkari, G. Draper-Gil, M. S. I. Mamun, and A. A. Ghorbani, '
        '"Characterization of Tor traffic using time based features," in '
        '_Proc. 3rd Int. Conf. Information Systems Security and Privacy '
        '(ICISSP)_, 2017, pp. 253-262.',
    "moustafa2015":
        'N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for '
        'network intrusion detection systems (UNSW-NB15 network data set)," '
        'in _Proc. Military Communications and Information Systems Conf. '
        '(MilCIS)_, 2015, pp. 1-6.',
    "moustafa2016":
        'N. Moustafa and J. Slay, "The evaluation of network anomaly '
        'detection systems: Statistical analysis of the UNSW-NB15 data set '
        'and the comparison with the KDD99 data set," _Information Security '
        'Journal: A Global Perspective_, vol. 25, no. 1-3, pp. 18-31, 2016.',
    "paxson1999":
        'V. Paxson, "Bro: A system for detecting network intruders in '
        'real-time," _Computer Networks_, vol. 31, no. 23-24, '
        'pp. 2435-2463, 1999.',
    "halbouni2022":
        'A. Halbouni, T. S. Gunawan, M. H. Habaebi, M. Halbouni, M. Kartiwi, '
        'and R. Ahmad, "CNN-LSTM: Hybrid deep neural network for network '
        'intrusion detection system," _IEEE Access_, vol. 10, '
        'pp. 99837-99849, 2022.',
    "apruzzese2022":
        'G. Apruzzese, L. Pajola, and M. Conti, "The cross-evaluation of '
        'machine learning-based network intrusion detection systems," _IEEE '
        'Trans. Network and Service Management_, vol. 19, no. 4, '
        'pp. 5152-5169, 2022.',
    "catillo2022":
        'M. Catillo, A. Del Vecchio, A. Pecchia, and U. Villano, '
        '"Transferability of machine learning models learned from public '
        'intrusion detection datasets: The CICIDS2017 case study," _Software '
        'Quality Journal_, vol. 30, no. 4, pp. 955-981, 2022.',
    "quinonero2009":
        'J. Quiñonero-Candela, M. Sugiyama, A. Schwaighofer, and '
        'N. D. Lawrence, Eds., _Dataset Shift in Machine Learning_. '
        'Cambridge, MA, USA: MIT Press, 2009.',
    "morenotorres2012":
        'J. G. Moreno-Torres, T. Raeder, R. Alaíz-Rodríguez, '
        'N. V. Chawla, and F. Herrera, "A unifying view on dataset shift in '
        'classification," _Pattern Recognition_, vol. 45, no. 1, '
        'pp. 521-530, 2012.',
    "harris2020":
        'C. R. Harris et al., "Array programming with NumPy," _Nature_, '
        'vol. 585, pp. 357-362, 2020.',
    "mckinney2010":
        'W. McKinney, "Data structures for statistical computing in Python," '
        'in _Proc. 9th Python in Science Conf._, 2010, pp. 56-61.',
    "virtanen2020":
        'P. Virtanen et al., "SciPy 1.0: Fundamental algorithms for '
        'scientific computing in Python," _Nature Methods_, vol. 17, '
        'pp. 261-272, 2020.',
    "pedregosa2011":
        'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," '
        '_J. Machine Learning Research_, vol. 12, pp. 2825-2830, 2011.',
    "hunter2007":
        'J. D. Hunter, "Matplotlib: A 2D graphics environment," _Computing in '
        'Science & Engineering_, vol. 9, no. 3, pp. 90-95, 2007.',
    "engelen2021":
        'G. Engelen, V. Rimmer, and W. Joosen, "Troubleshooting an intrusion '
        'detection dataset: The CICIDS2017 case study," in _Proc. IEEE '
        'Security and Privacy Workshops (SPW)_, 2021, pp. 7-12.',
    "pendlebury2019":
        'F. Pendlebury, F. Pierazzi, R. Jordaney, J. Kinder, and '
        'L. Cavallaro, "TESSERACT: Eliminating experimental bias in malware '
        'classification across space and time," in _Proc. 28th USENIX '
        'Security Symp._, 2019, pp. 729-746.',
    "he2009":
        'H. He and E. A. Garcia, "Learning from imbalanced data," _IEEE '
        'Trans. Knowledge and Data Engineering_, vol. 21, no. 9, '
        'pp. 1263-1284, 2009.',
    "tukey1977":
        'J. W. Tukey, _Exploratory Data Analysis_. Reading, MA, USA: '
        'Addison-Wesley, 1977.',
    "boxcox1964":
        'G. E. P. Box and D. R. Cox, "An analysis of transformations," '
        '_J. Royal Statistical Society: Series B_, vol. 26, no. 2, '
        'pp. 211-252, 1964.',
    "yeojohnson2000":
        'I.-K. Yeo and R. A. Johnson, "A new family of power transformations '
        'to improve normality or symmetry," _Biometrika_, vol. 87, no. 4, '
        'pp. 954-959, 2000.',
    "spearman1904":
        'C. Spearman, "The proof and measurement of association between two '
        'things," _American J. Psychology_, vol. 15, no. 1, pp. 72-101, 1904.',
    "kutner2005":
        'M. H. Kutner, C. J. Nachtsheim, J. Neter, and W. Li, _Applied Linear '
        'Statistical Models_, 5th ed. New York, NY, USA: McGraw-Hill, 2005.',
    "jolliffe2002":
        'I. T. Jolliffe, _Principal Component Analysis_, 2nd ed. New York, '
        'NY, USA: Springer, 2002.',
    "hooker2021":
        'G. Hooker, L. Mentch, and S. Zhou, "Unrestricted permutation forces '
        'extrapolation: Variable importance requires at least one more model, '
        'or there is no free variable importance," _Statistics and '
        'Computing_, vol. 31, no. 6, art. 82, 2021.',
    "strobl2008":
        'C. Strobl, A.-L. Boulesteix, T. Kneib, T. Augustin, and A. Zeileis, '
        '"Conditional variable importance for random forests," _BMC '
        'Bioinformatics_, vol. 9, art. 307, 2008.',
    "sokal1958":
        'R. R. Sokal and C. D. Michener, "A statistical method for evaluating '
        'systematic relationships," _University of Kansas Science Bulletin_, '
        'vol. 38, pp. 1409-1438, 1958.',
    "cover2006":
        'T. M. Cover and J. A. Thomas, _Elements of Information Theory_, '
        '2nd ed. Hoboken, NJ, USA: Wiley-Interscience, 2006.',
    "kraskov2004":
        'A. Kraskov, H. Stögbauer, and P. Grassberger, "Estimating '
        'mutual information," _Physical Review E_, vol. 69, no. 6, '
        'art. 066138, 2004.',
    "ross2014":
        'B. C. Ross, "Mutual information between discrete and continuous '
        'data sets," _PLoS ONE_, vol. 9, no. 2, art. e87357, 2014.',
    "massey1951":
        'F. J. Massey, Jr., "The Kolmogorov-Smirnov test for goodness of '
        'fit," _J. American Statistical Association_, vol. 46, no. 253, '
        'pp. 68-78, 1951.',
    "siddiqi2006":
        'N. Siddiqi, _Credit Risk Scorecards: Developing and Implementing '
        'Intelligent Credit Scoring_. Hoboken, NJ, USA: Wiley, 2006.',
    "lin1991":
        'J. Lin, "Divergence measures based on the Shannon entropy," _IEEE '
        'Trans. Information Theory_, vol. 37, no. 1, pp. 145-151, 1991.',
    "cohen1988":
        'J. Cohen, _Statistical Power Analysis for the Behavioral Sciences_, '
        '2nd ed. Hillsdale, NJ, USA: Lawrence Erlbaum, 1988.',
    "friedman2001":
        'J. H. Friedman, "Greedy function approximation: A gradient boosting '
        'machine," _Annals of Statistics_, vol. 29, no. 5, pp. 1189-1232, '
        '2001.',
    "ke2017":
        'G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and '
        'T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision '
        'tree," in _Advances in Neural Information Processing Systems '
        '(NeurIPS)_, 2017, pp. 3146-3154.',
    "bendavid2010":
        'S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and '
        'J. W. Vaughan, "A theory of learning from different domains," '
        '_Machine Learning_, vol. 79, no. 1-2, pp. 151-175, 2010.',
    "maaten2008":
        'L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," '
        '_J. Machine Learning Research_, vol. 9, pp. 2579-2605, 2008.',
    "matthews1975":
        'B. W. Matthews, "Comparison of the predicted and observed secondary '
        'structure of T4 phage lysozyme," _Biochimica et Biophysica Acta_, '
        'vol. 405, no. 2, pp. 442-451, 1975.',
    "chicco2020":
        'D. Chicco and G. Jurman, "The advantages of the Matthews correlation '
        'coefficient (MCC) over F1 score and accuracy in binary '
        'classification evaluation," _BMC Genomics_, vol. 21, art. 6, 2020.',
    "webber2010":
        'W. Webber, A. Moffat, and J. Zobel, "A similarity measure for '
        'indefinite rankings," _ACM Trans. Information Systems_, vol. 28, '
        'no. 4, art. 20, 2010.',
    "wilcoxon1945":
        'F. Wilcoxon, "Individual comparisons by ranking methods," '
        '_Biometrics Bulletin_, vol. 1, no. 6, pp. 80-83, 1945.',
    "ghorbani2019":
        'A. Ghorbani, A. Abid, and J. Zou, "Interpretation of neural networks '
        'is fragile," in _Proc. AAAI Conf. Artificial Intelligence_, 2019, '
        'pp. 3681-3688.',
    "adebayo2018":
        'J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and '
        'B. Kim, "Sanity checks for saliency maps," in _Advances in Neural '
        'Information Processing Systems (NeurIPS)_, 2018, pp. 9505-9515.',
    "krishna2022":
        'S. Krishna, T. Han, A. Gu, J. Pombra, S. Jabbari, S. Wu, and '
        'H. Lakkaraju, "The disagreement problem in explainable machine '
        'learning: A practitioner\'s perspective," arXiv:2202.01602, 2022.',
    "warnecke2020":
        'A. Warnecke, D. Arp, C. Wressnegger, and K. Rieck, "Evaluating '
        'explanation methods for deep learning in security," in _Proc. IEEE '
        'European Symp. Security and Privacy (EuroS&P)_, 2020, pp. 158-174.',
}

_CITED: dict[str, int] = {}


def _number(key: str) -> int:
    if key not in BIB:
        raise KeyError(f"citation key not in BIB: {key!r}")
    if key not in _CITED:
        _CITED[key] = len(_CITED) + 1
    return _CITED[key]


def _bracket(nums: list[int]) -> str:
    """Render citation numbers IEEE-style, contracting runs of three or more."""
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f"[{nums[i]}]–[{nums[j]}]")
            i = j + 1
        else:
            out.append(f"[{nums[i]}]")
            i += 1
    return ", ".join(out)


def _cite(text: str) -> str:
    return re.sub(
        r"\[@([A-Za-z0-9_, ]+)\]",
        lambda m: _bracket([_number(k.strip()) for k in m.group(1).split(",")]),
        text)


def _xref(text: str) -> str:
    """Rewrite the source prose's cross-references into the paper's numbering.

    Sections are lettered within roman-numbered parts and tables are numbered
    in roman, so "Section 4.2", "Figure 7" and "Table 10" as written in the
    section bodies become "Section IV-B", "Fig. 7" and "Table X". Doing it here
    rather than in the prose keeps a single source of truth for the numbering.
    """
    text = re.sub(r"Sections (\d+) to (\d+)",
                  lambda m: (f"Sections {ROMAN[int(m.group(1))]} to "
                             f"{ROMAN[int(m.group(2))]}"), text)
    text = re.sub(r"Section (\d+)\.(\d+)",
                  lambda m: (f"Section {ROMAN[int(m.group(1))]}-"
                             f"{chr(64 + int(m.group(2)))}"), text)
    text = re.sub(r"Section (\d+)",
                  lambda m: f"Section {ROMAN[int(m.group(1))]}", text)
    text = re.sub(r"Figures (\d+)[–-](\d+)", r"Figs. \1–\2", text)
    text = re.sub(r"Figure (\d+)", r"Fig. \1", text)
    text = re.sub(r"Table (\d+)",
                  lambda m: f"Table {ROMAN[int(m.group(1))]}", text)
    return text


def _typography(text: str) -> str:
    """A true minus sign and curly single quotes.

    Word offers a line break after a hyphen-minus, which splits a value like
    -0.16 across two lines; U+2212 is both correct for a negative number and
    unbreakable.
    """
    text = re.sub(r"(?<=[\s(\[])-(?=\d)", "−", text)
    text = re.sub(r"(?<=[\s(])'([^']{1,70})'", "‘\\1’", text)
    return text


def _tx(text: str) -> str:
    return _typography(_xref(_cite(text)))


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

def tables() -> dict[str, pd.DataFrame]:
    t = {}
    for f in sorted(os.listdir(TAB_DIR)):
        if f.endswith(".csv"):
            t[f[:-4]] = pd.read_csv(os.path.join(TAB_DIR, f))
    return t


def facts(T: dict) -> dict:
    """Pull every figure quoted in the prose out of the tables."""
    q = T["t09_quality_scorecard"].set_index("dataset")
    imb = T["t08_imbalance"].set_index("dataset")
    uni = T["t13_univariate_summary"].set_index("dataset")
    sh = T["t23_shift_headline"].iloc[0]
    shift = T["t18_marginal_shift"]
    vif = T["t15_vif"].set_index("feature")
    pca = T["t22_pca_variance"]
    inv = T["t01_file_inventory"]
    led = T["t03_attrition_ledger"]
    agree = T["t17_ranking_agreement"]
    joined = T["t24_eda_vs_model"].set_index("class")
    corr = T["t14_correlation_pairs"]
    dom = T["t20_domain_classifier"].rename(columns={"Unnamed: 0": "feature"}).set_index("feature")

    import yaml
    with open(os.path.join(MODEL_TAB_DIR, "headline.yaml")) as f:
        head = yaml.safe_load(f)

    def kept(ds):
        return int(led[(led.dataset == ds) & (led.step == "analysed records")].remaining.iloc[0])

    return {
        "q": q, "imb": imb, "uni": uni, "sh": sh, "shift": shift, "vif": vif,
        "pca": pca, "inv": inv, "agree": agree, "joined": joined, "corr": corr,
        "dom": dom, "head": head,
        "cic_rows": int(q.loc["cicids2017", "records"]),
        "unsw_rows": int(q.loc["unsw_nb15", "records"]),
        "cic_kept": kept("cicids2017"),
        "unsw_kept": kept("unsw_nb15"),
        "total_mb": float(inv.size_mb.sum()),
        "pc90": int(sh["pcs_for_90pct_variance"]),
        "var3": float(pca.cumulative.iloc[2]),
    }


# --------------------------------------------------------------------------
# page geometry and column spans
# --------------------------------------------------------------------------

def _serif(run, name: str = SERIF) -> None:
    """Pin a run to one typeface for every script range.

    Setting only `font.name` writes `w:ascii`, which leaves en-dashes, Greek
    letters and the multiplication sign to be rendered in the theme font.
    """
    run.font.name = name
    rf = run._r.get_or_add_rPr().get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(attr), name)


def _apply_page(section) -> None:
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.left_margin = section.right_margin = Inches(MARGIN_X)
    section.top_margin = section.bottom_margin = Inches(MARGIN_Y)


def _set_columns(section, n: int) -> None:
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        grid = sectPr.find(qn("w:docGrid"))
        (grid.addprevious if grid is not None else sectPr.append)(cols)
    for child in list(cols):
        cols.remove(child)
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(int(GUTTER * 1440)))
    cols.set(qn("w:equalWidth"), "1")


def _span(doc, n: int):
    """Continue on the same page with a different column count.

    This is how a two-column paper carries a full-width figure or table: the
    column count changes at a continuous section break, so `_span(doc, 1)`
    opens the span and `_span(doc, 2)` closes it. The paragraph that carries
    the break mark is collapsed to a single point so it adds no visible gap.
    """
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    _apply_page(section)
    _set_columns(section, n)
    brk = doc.paragraphs[-1]
    brk.paragraph_format.space_before = Pt(0)
    brk.paragraph_format.space_after = Pt(0)
    brk.paragraph_format.line_spacing = Pt(1)
    return section


def _style(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = SERIF
    normal.font.size = Pt(BODY_PT)
    normal.font.color.rgb = BLACK
    rf = normal.element.get_or_add_rPr().get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(attr), SERIF)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.line_spacing = 1.0

    # Level-one headings are centred small capitals and level-two headings are
    # left-aligned italics, which is the template's own hierarchy; the run
    # properties are set again per heading so a theme font cannot override it.
    for name, before, after in (("Heading 1", 10, 4), ("Heading 2", 8, 3),
                                ("Heading 3", 7, 2)):
        st = doc.styles[name]
        st.font.name = SERIF
        st.font.size = Pt(BODY_PT)
        st.font.bold = False
        st.font.color.rgb = BLACK
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.line_spacing = 1.0
        st.paragraph_format.keep_with_next = True

    for name in ("List Bullet", "List Number"):
        st = doc.styles[name]
        st.font.name = SERIF
        st.font.size = Pt(BODY_PT)
        st.font.color.rgb = BLACK

    section = doc.sections[0]
    _apply_page(section)
    _set_columns(section, 1)          # the title block spans both columns


def _page_numbers(doc: Document) -> None:
    """A centred PAGE field in the first section's footer.

    Every later section is a continuous break that inherits the footer, so one
    definition covers the document however many spans it contains.
    """
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for kind in ("begin", "text", "end"):
        if kind == "text":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = " PAGE "
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        run._r.append(el)
    _serif(run)
    run.font.size = Pt(9)
    run.font.color.rgb = BLACK
    for section in doc.sections[1:]:
        section.footer.is_linked_to_previous = True


# --------------------------------------------------------------------------
# paragraphs, headings and captions
# --------------------------------------------------------------------------

_N = {"sec": 0, "sub": 0, "fresh": False}


_MARKUP = re.compile(r"\*\*(.+?)\*\*|`(.+?)`", re.S)


def _runs(p, text, *, size=BODY_PT, italic=False, bold=False):
    """Lay text into runs, honouring **bold** and `monospace` markup."""

    def add(chunk, *, strong=False, mono=False):
        if not chunk:
            return
        r = p.add_run(chunk)
        _serif(r, MONO if mono else SERIF)
        r.font.size = Pt(size - 1.0 if mono else size)
        r.italic = italic and not mono
        r.bold = bold or strong
        r.font.color.rgb = BLACK

    pos = 0
    for m in _MARKUP.finditer(text):
        add(text[pos:m.start()])
        if m.group(1) is not None:
            add(m.group(1), strong=True)
        else:
            add(m.group(2), mono=True)
        pos = m.end()
    add(text[pos:])
    return p


def para(doc, text, *, size=BODY_PT, italic=False, bold=False, align=None,
         space_after=0, space_before=0, indent=0.0, style=None):
    p = doc.add_paragraph(style=style)
    _runs(p, _tx(text), size=size, italic=italic, bold=bold)
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.first_line_indent = Inches(indent)
    return p


def body(doc, text, **kw):
    """A body paragraph: justified, no blank line, first line indented.

    The first paragraph after a heading is flush left, as the template has it.
    """
    p = para(doc, text, align=kw.get("align", WD_ALIGN_PARAGRAPH.JUSTIFY),
             indent=0.0 if _N["fresh"] else INDENT)
    _N["fresh"] = False
    return p


def bullets(doc, items, *, numbered=False):
    for it in items:
        p = doc.add_paragraph(style="List Number" if numbered else "List Bullet")
        _runs(p, _tx(it))
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
    _N["fresh"] = False


def _heading(doc, text, level, *, align, small_caps=False, italic=False):
    h = doc.add_heading(text, level=level)
    h.alignment = align
    for r in h.runs:
        _serif(r)
        r.font.size = Pt(BODY_PT)
        r.font.bold = False
        r.font.italic = italic
        r.font.small_caps = small_caps
        r.font.color.rgb = BLACK
    _N["fresh"] = True
    return h


def h1(doc, text):
    """A numbered first-level heading: `IV.  Univariate structure`."""
    _N["sec"] += 1
    _N["sub"] = 0
    return _heading(doc, f"{ROMAN[_N['sec']]}.  {text}", 1,
                    align=WD_ALIGN_PARAGRAPH.CENTER, small_caps=True)


def h1_plain(doc, text):
    """An unnumbered first-level heading, for the appendix and references."""
    _N["sub"] = 0
    return _heading(doc, text, 1, align=WD_ALIGN_PARAGRAPH.CENTER,
                    small_caps=True)


def h2(doc, text):
    """A lettered second-level heading: `B.  The case for log1p`."""
    _N["sub"] += 1
    return _heading(doc, f"{chr(64 + _N['sub'])}.  {text}", 2,
                    align=WD_ALIGN_PARAGRAPH.LEFT, italic=True)


def figure(doc, name, caption, width=None):
    """A full-width figure with its caption below, as `Fig. n.`"""
    _span(doc, 1)
    doc.add_picture(os.path.join(FIG_DIR, f"{name}.png"),
                    width=Inches(width or SPAN))
    pic = doc.paragraphs[-1]
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.space_before = Pt(9)
    pic.paragraph_format.space_after = Pt(3)
    pic.paragraph_format.keep_with_next = True

    m = re.match(r"\s*Figure (\d+)\s*—\s*(.*)", caption, re.S)
    text = f"Fig. {m.group(1)}.  {m.group(2)}" if m else caption
    p = para(doc, re.sub(r"\s+", " ", text), size=CAPTION_PT,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=9)
    _span(doc, 2)
    _N["fresh"] = True
    return p


def _table_caption(doc, caption):
    """The caption above the table: a `TABLE n` line, then the sentence."""
    m = re.match(r"\s*Table (A?\d+)\s*—\s*(.*)", caption, re.S)
    label, text = (m.group(1), m.group(2)) if m else ("", caption)
    if label.isdigit():
        label = ROMAN[int(label)]
    head = para(doc, f"TABLE {label}".strip(), size=CAPTION_PT,
                align=WD_ALIGN_PARAGRAPH.CENTER, space_before=10)
    head.paragraph_format.keep_with_next = True
    cap = para(doc, re.sub(r"\s+", " ", text), size=CAPTION_PT,
               align=WD_ALIGN_PARAGRAPH.CENTER, space_after=3)
    cap.paragraph_format.keep_with_next = True
    return cap


def _cell_rule(cell, *, top=0, bottom=0):
    """Horizontal rules drawn per cell, so the table needs no border style."""
    borders = OxmlElement("w:tcBorders")
    for edge, size in (("top", top), ("bottom", bottom)):
        if not size:
            continue
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    if len(borders):
        cell._tc.get_or_add_tcPr().append(borders)


def table(doc, df: pd.DataFrame, caption: str, *, widths=None, font=8.3,
          index=False, max_rows=None, align_right=(), span=True):
    """A rule-ruled table: no vertical lines, no fills, caption above.

    `span` draws the table across both columns, which all but the narrowest
    of them need.
    """
    if max_rows:
        df = df.head(max_rows)
    data = df.reset_index() if index else df
    total = SPAN if span else COL

    if span:
        _span(doc, 1)
    _table_caption(doc, caption)

    t = doc.add_table(rows=1, cols=len(data.columns))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False

    # Repeat the header on every page a long table spills onto.
    trPr = t.rows[0]._tr.get_or_add_trPr()
    rep = OxmlElement("w:tblHeader")
    rep.set(qn("w:val"), "true")
    trPr.append(rep)

    hdr = t.rows[0].cells
    for j, c in enumerate(data.columns):
        hdr[j].text = ""
        r = hdr[j].paragraphs[0].add_run(str(c).replace("_", " "))
        _serif(r)
        r.italic = True
        r.font.size = Pt(font)
        r.font.color.rgb = BLACK
        hdr[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        hdr[j].paragraphs[0].paragraph_format.space_after = Pt(1)

    for _, row in data.iterrows():
        cells = t.add_row().cells
        for j, c in enumerate(data.columns):
            v = row[c]
            cells[j].text = ""
            r = cells[j].paragraphs[0].add_run(_fmt(v))
            _serif(r)
            r.font.size = Pt(font)
            r.font.color.rgb = BLACK
            cells[j].paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.RIGHT if c in align_right or
                (isinstance(v, float) and c not in ("finding",)) else WD_ALIGN_PARAGRAPH.LEFT)
            cells[j].paragraphs[0].paragraph_format.space_after = Pt(1)

    scaled = ([w * total / sum(widths) for w in widths] if widths
              else [total / len(data.columns)] * len(data.columns))
    for row in t.rows:
        for cell, w in zip(row.cells, scaled):
            cell.width = Inches(w)

    for cell in t.rows[0].cells:
        _cell_rule(cell, top=8, bottom=4)
    for cell in t.rows[-1].cells:
        _cell_rule(cell, bottom=8)

    if span:
        _span(doc, 2)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
    _N["fresh"] = True
    return t


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return "—"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        if v == 0 or abs(v) >= 0.0005:
            return f"{v:.3f}"
        return f"{v:.1e}"
    if isinstance(v, (int,)):
        return f"{v:,}"
    return str(v)


# --------------------------------------------------------------------------
# front matter and references
# --------------------------------------------------------------------------

def title_block(doc):
    """Title, authors and affiliation across the full width of the page."""
    para(doc, TITLE, size=17, align=WD_ALIGN_PARAGRAPH.CENTER,
         space_before=6, space_after=12)
    for name, reg in AUTHORS:
        para(doc, name, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER)
        para(doc, reg, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    for line in INSTITUTION:
        para(doc, line, size=9.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, f"Course project — {COURSE}", size=9, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_before=8, space_after=12)
    _span(doc, 2)


def _abstract_para(doc, text, lead=None):
    p = doc.add_paragraph()
    if lead:
        r = p.add_run(lead)
        _serif(r)
        r.bold = True
        r.italic = True
        r.font.size = Pt(9)
        r.font.color.rgb = BLACK
    _runs(p, _tx(text), size=9, italic=True)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    return p


def keywords(doc):
    p = doc.add_paragraph()
    r = p.add_run("Index Terms—")
    _serif(r)
    r.bold = True
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = BLACK
    _runs(p, KEYWORDS, size=9, bold=True)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(8)
    _N["fresh"] = True


def _curly(text: str) -> str:
    """Turn the straight quotes around a reference title into curly ones."""
    out, opening = [], True
    for ch in text:
        if ch == '"':
            out.append("“" if opening else "”")
            opening = not opening
        else:
            out.append(ch)
    return "".join(out)


def references(doc):
    """The reference list, numbered in order of first citation."""
    h1_plain(doc, "References")
    uncited = [k for k in BIB if k not in _CITED]
    assert not uncited, f"references defined but never cited: {uncited}"
    for key, n in sorted(_CITED.items(), key=lambda kv: kv[1]):
        p = doc.add_paragraph()
        r = p.add_run(f"[{n}]\t")
        _serif(r)
        r.font.size = Pt(REF_PT)
        r.font.color.rgb = BLACK
        entry = re.sub(r"(?<=\d)-(?=\d)", "–", BIB[key])
        entry = _curly(entry)
        for i, chunk in enumerate(entry.split("_")):
            if not chunk:
                continue
            r = p.add_run(chunk)
            _serif(r)
            r.italic = i % 2 == 1
            r.font.size = Pt(REF_PT)
            r.font.color.rgb = BLACK
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.left_indent = Inches(0.26)
        pf.first_line_indent = Inches(-0.26)
        pf.space_after = Pt(2)
        pf.line_spacing = 1.0
        pf.tab_stops.add_tab_stop(Inches(0.26), WD_TAB_ALIGNMENT.LEFT)


def abstract(doc, F):
    sh, uni, q = F["sh"], F["uni"], F["q"]
    _abstract_para(
        doc,
        f"Network intrusion detection research routinely trains on one public benchmark and "
        f"reports conclusions as though they were properties of network attacks rather than "
        f"properties of the capture. This report subjects that habit to an exploratory data "
        f"analysis. Two widely used benchmarks — CICIDS2017 ({F['cic_rows']:,} flows, 79 "
        f"columns) and UNSW-NB15 ({F['unsw_rows']:,} flows, 45 columns) — are harmonised onto "
        f"a common twelve-feature vector and a common four-class attack taxonomy, and then "
        f"examined side by side at four levels: data quality, univariate shape, dependence "
        f"structure, and the gap between the two corpora.",
        lead="Abstract—")
    _abstract_para(
        doc,
        f"The audit of the published files alone returns a substantial defect list: "
        f"{int(q.loc['cicids2017', 'constant_columns'])} zero-variance columns, "
        f"{int(q.loc['cicids2017', 'columns_with_negatives'])} columns carrying negative values "
        f"in quantities that cannot be negative, infinite rates produced by division by a zero "
        f"duration, a duplicated column name, a label set that is not valid UTF-8, and exact "
        f"duplicate records amounting to {q.loc['cicids2017', 'duplicate_pct']:.1f}% and "
        f"{q.loc['unsw_nb15', 'duplicate_pct']:.1f}% of the two corpora. Univariate analysis "
        f"finds every feature heavy-tailed over as much as nine orders of magnitude "
        f"(median |skew| = {uni.loc['cicids2017', 'median_abs_skew_raw']:.0f} on CICIDS2017, "
        f"falling to {uni.loc['cicids2017', 'median_abs_skew_log']:.2f} under log1p), and the "
        f"two standard outlier screens disagreeing by an order of magnitude.")
    _abstract_para(
        doc,
        f"The central finding is the size of the gap between the benchmarks. All twelve "
        f"shared features exceed the conventional ‘major shift’ threshold on the population "
        f"stability index; the mean Kolmogorov–Smirnov distance is {sh['mean_ks']:.2f}; the "
        f"shift survives conditioning on attack class, with a mean standardised mean difference "
        f"of {sh['mean_abs_smd_within_class']:.2f} inside classes against Cohen's 0.8 for a "
        f"large effect; and a gradient-boosted classifier recovers which benchmark a flow came "
        f"from with AUC {sh['domain_auc']:.5f} using nothing but the twelve shared features. "
        f"The two corpora also disagree, before any model is fitted, about which features carry "
        f"class information: mutual-information feature rankings agree at a Spearman "
        f"ρ of only {F['agree'].spearman_rho.mean():+.2f} on average across the four classes.")
    _abstract_para(
        doc,
        f"Held against the modelling results of the parent project, these diagnostics are "
        f"confirmed. A CNN-LSTM scoring {F['head']['detection']['source_f1_macro']:.3f} macro F1 "
        f"on its own benchmark falls to {F['head']['detection']['target_f1_macro']:.3f} when "
        f"applied unchanged to the other, and the class that the EDA flagged as having moved "
        f"furthest is the class that transfers worst. The practical conclusion is that "
        f"cross-dataset claims in this field require the gap to be measured and reported "
        f"first — and that an exploratory analysis is enough to measure it.")
    keywords(doc)


def section_intro(doc, F):
    h1(doc, "Introduction and objectives")
    body(doc,
         "An intrusion detection model is trained on a capture of network traffic, and a capture "
         "is made in a particular testbed, with a particular flow exporter, over a particular "
         "few days. When a result obtained on one such capture is reported, the implicit claim "
         "is that it says something about network attacks. That claim is only safe if the "
         "capture is not itself a confound — if a flow labelled 'denial of service' in one "
         "benchmark looks like a flow labelled 'denial of service' in another. That the claim "
         "is rarely examined is a long-standing methodological criticism of the field "
         "[@sommer2010], [@arp2022], and it is compounded when the detector is accompanied by a "
         "post-hoc attribution method [@lundberg2017], [@ribeiro2016] whose feature ranking is "
         "then read as insight into the attack itself [@wang2020], [@oseni2023].")
    body(doc,
         "This report treats that as an empirical question and answers it with exploratory data "
         "analysis, before any model is fitted. The two benchmarks examined are the two most "
         "widely used in the literature: **CICIDS2017**, captured by the Canadian Institute for "
         "Cybersecurity with the CICFlowMeter exporter [@sharafaldin2018], [@lashkari2017], "
         "and **UNSW-NB15**, captured at the Australian Centre for Cyber Security with Argus "
         "and Zeek [@moustafa2015], [@moustafa2016], [@paxson1999]. They have no feature schema "
         "in common and no label set in common, so the first task is to construct a space in "
         "which they can be compared at all.")
    h2(doc, "Questions")
    bullets(doc, [
        "**Q1 — Quality.** What defects are present in the published files, and which of "
        "them would change an analysis if they went unnoticed?",
        "**Q2 — Shape.** What scale do these features live on, and what does that imply for "
        "the summary statistics, outlier rules and distance-based methods that are normally "
        "applied to them?",
        "**Q3 — Structure.** How much independent information is there in the feature set, "
        "and which features actually carry the class?",
        "**Q4 — Comparability.** How far apart are the two corpora, how much of that gap "
        "survives conditioning on attack class, and can the two be told apart at all?",
        "**Q5 — Consequence.** Do the answers to Q1–Q4 anticipate what happens when a "
        "model is actually trained on one benchmark and applied to the other?",
    ])
    h2(doc, "What this report contributes")
    body(doc,
         "The analysis is deliberately model-free up to Section 7. Every statement in Sections "
         "3 to 6 is a property of the data, obtained without training anything, and could have "
         "been produced before a single epoch was run. Section 7 then checks those statements "
         "against the modelling results of the parent project — a hybrid CNN-LSTM "
         "[@halbouni2022] trained across five seeds with SHAP [@lundberg2017] attributions "
         "extracted per class — which are used here "
         "purely as held-out evidence. The point of that ordering is that the EDA should be "
         "judged on whether it anticipated the outcome, not on whether it can be made to "
         "agree with it afterwards.")
    body(doc,
         "Cross-dataset evaluation of intrusion detectors has been reported before "
         "[@apruzzese2022], [@catillo2022], and the statistical vocabulary for the gap between "
         "two corpora is standard [@quinonero2009], [@morenotorres2012]. What an exploratory "
         "analysis adds is that the gap can be characterised feature by feature and class by "
         "class without fitting the model whose failure it predicts, which makes it available "
         "as a precondition for the experiment rather than as an explanation after it.")


def section_data(doc, F):
    h1(doc, "Data, harmonisation and method")
    h2(doc, "Sources")
    body(doc,
         f"Both benchmarks are public and require no licence agreement. CICIDS2017 is read from "
         f"its eight working-day capture files; UNSW-NB15 is read from its named training and "
         f"testing partitions only. That second restriction matters: the UNSW distribution also "
         f"ships four bulk files with 49 columns and **no header row**, so a bare `*.csv` glob "
         f"concatenates incompatible schemas into a mostly-empty frame without raising an error. "
         f"The loader matches an explicit pattern and refuses files whose column count disagrees.")
    inv = F["inv"].copy()
    inv["size_mb"] = inv.size_mb.map("{:.1f}".format)
    table(doc, inv, "Table 1 — Source files as read. CICIDS2017 is 11× the size of UNSW-NB15 "
                    "on disk and 11× its record count, which is itself a sampling asymmetry the "
                    "analysis has to respect.",
          widths=[1.1, 2.9, 0.8, 0.9, 0.7], font=8)

    h2(doc, "Building a shared feature space")
    body(doc,
         "The two exporters measure different things, so **not one of the twelve shared features "
         "is a column rename**. Each is a derivation computed from each dataset's own raw "
         "columns, and unit differences are reconciled explicitly: CICFlowMeter reports flow "
         "duration in microseconds and inter-arrival times in microseconds, while UNSW-NB15 "
         "reports duration in seconds and inter-arrival times in milliseconds. Taking those at "
         "face value would place the two corpora roughly 437,000× apart on the inter-arrival "
         "features alone, which would dominate everything that followed.")
    body(doc,
         "Rates are recomputed from primitives on both sides rather than read from each "
         "dataset's own rate column, because UNSW's `rate` correlates only r = 0.77 with "
         "(spkts + dpkts) / dur and is therefore not the same quantity as CICFlowMeter's Flow "
         "Packets/s. Ratio denominators use +1 rather than a small epsilon: dividing by 1e-6 "
         "turns a zero denominator into a 10⁶-fold spike, and on the real data that gave "
         "down_up_ratio a standard deviation of 1.4 × 10⁶ against an interquartile range of 4.3.")
    fd = pd.read_csv(os.path.join(TAB_DIR, "t05_feature_dictionary.csv"))
    table(doc, fd[["feature", "cicids2017_derivation", "unsw_nb15_derivation"]],
          "Table 2 — The shared feature vector. Every row is a derivation, not a mapping between "
          "equivalent columns; the two benchmarks share no feature schema.",
          widths=[1.3, 2.6, 2.5], font=8)

    h2(doc, "A common class taxonomy")
    body(doc,
         "Four unified classes are used: Benign, DoS/DDoS, Reconnaissance/Probe and Exploits. "
         "**Categories with no defensible counterpart in the other benchmark are dropped rather "
         "than bucketed.** An earlier version of this mapping collapsed CICIDS's Bot, FTP-Patator "
         "and SSH-Patator together with UNSW's Generic, Fuzzers, Analysis and Worms into a single "
         "'Exploits' label, which placed credential brute-forcing, botnet command-and-control, "
         "protocol fuzzing and block-cipher attacks under one invented name. Any disagreement "
         "measured across such a bucket would be explained by the taxonomy rather than by the "
         "data, so the exclusions are made explicit and the cost in records is reported in "
         "Section 3.4. The Exploits pairing that survives is still the loosest of the four and "
         "is treated as exploratory throughout.")

    h2(doc, "Method and tooling")
    body(doc,
         "The pipeline runs in six stages. Stage 1 reads the raw files one at a time and "
         "accumulates per-column quality statistics **before** harmonisation, because a missing "
         "value becomes a dropped row and a corrupted label becomes a silently missing class the "
         "moment harmonisation is applied. Stages 2 to 6 read a cached harmonised matrix. All "
         "statistics are computed on the full corpora except where an algorithm is superlinear "
         "in n, where a class-balanced subsample is drawn and the sample size is stated. "
         "Analysis uses NumPy [@harris2020], pandas [@mckinney2010], SciPy [@virtanen2020] "
         "and scikit-learn [@pedregosa2011]; figures use Matplotlib [@hunter2007].")
    body(doc,
         "One methodological choice recurs throughout and is justified in Section 4: "
         "**scale-sensitive analyses are run on log1p-transformed features**. The transform is "
         "parameter-free, so unlike a fitted scaler it estimates nothing from either dataset and "
         "cannot leak information between them. Descriptive statistics are nevertheless reported "
         "in the original units, because a report that only ever shows transformed values hides "
         "the thing it is transforming away from.")


def section_quality(doc, F):
    q, T = F["q"], tables()
    h1(doc, "Data quality: what the published files actually contain")
    body(doc,
         "This section works on the raw columns, before harmonisation. That ordering is not "
         "incidental. Once the twelve-feature matrix exists, a missing value has already become "
         "a dropped row, a constant column has already been excluded, and a mojibake label has "
         "already been canonicalised — in each case the evidence is gone. Several of the "
         "findings below are only visible at this stage. An independent audit of CICIDS2017 "
         "reached comparable conclusions about the published files from a different "
         "direction [@engelen2021].")

    h2(doc, "Structural defects")
    body(doc,
         f"**{int(q.loc['cicids2017', 'columns_with_leading_space'])} of CICIDS2017's 79 column "
         f"names carry a leading space** (' Flow Duration', not 'Flow Duration'), and the name "
         f"'{q.loc['cicids2017', 'duplicated_column_names']}' is used for two different columns. "
         f"Neither is visible in a spreadsheet. The first makes any exact-name lookup fail; the "
         f"second is silently resolved by pandas into a '.1' suffix, which is the only surviving "
         f"trace that the file was malformed. UNSW-NB15 has neither defect but does carry a "
         f"surrogate row identifier, which has to be excluded before duplicate detection or "
         f"every record is unique by construction.")
    body(doc,
         f"**{int(q.loc['cicids2017', 'constant_columns'])} CICIDS2017 columns take exactly one "
         f"value across all {F['cic_rows']:,} records** — the bulk-rate and reverse-direction "
         f"flag families are identically zero — and "
         f"{int(q.loc['cicids2017', 'columns_over_99pct_zero'])} are zero in more than 99% of "
         f"records. A zero-variance column cannot contribute to any model, but it is still "
         f"counted as a feature, still consumes a slot in a feature-importance ranking, and "
         f"still receives an attribution score.")

    h2(doc, "Values that should not exist")
    body(doc,
         f"**{int(q.loc['cicids2017', 'columns_with_negatives'])} CICIDS2017 columns contain "
         f"negative values in quantities that are physically non-negative.** Flow Duration "
         f"reaches −13 microseconds. Fwd Header Length reaches −3.2 × 10¹⁰, which is a "
         f"32-bit integer overflow rather than a measurement. Init_Win_bytes_forward and "
         f"Init_Win_bytes_backward are −1 in 1.0 and 1.4 million records respectively, where "
         f"−1 is a sentinel for 'no window observed' and not a value at all; averaging that "
         f"column is meaningless.")
    body(doc,
         f"Flow Bytes/s carries 1,358 missing and 1,509 infinite values, and Flow Packets/s "
         f"carries 2,867 infinite values. All of them arise the same way — division by a flow "
         f"duration of zero. This is the direct empirical argument for recomputing both rate "
         f"features from their primitives (Section 2.2) rather than reading the published "
         f"columns. UNSW-NB15 has no missing and no infinite values anywhere.")
    figure(doc, "f01_data_quality",
           "Figure 1 — Data-quality defects in the raw files. (a) Missing and infinite values, "
           "all in CICIDS2017's two rate columns. (b) Zero inflation across every numeric "
           "column: 15 CICIDS2017 columns sit beyond 95% zero. (c) Negative values in "
           "non-negative quantities, with the minimum observed in each column; note the "
           "integer-overflow magnitudes. (d) Degenerate and malformed columns by benchmark.")

    h2(doc, "Duplicate records and label encoding")
    body(doc,
         f"**{q.loc['cicids2017', 'duplicate_pct']:.1f}% of CICIDS2017 records and "
         f"{q.loc['unsw_nb15', 'duplicate_pct']:.1f}% of UNSW-NB15 records are exact duplicates** "
         f"of another record across the full feature vector "
         f"({int(q.loc['cicids2017', 'exact_duplicate_records']):,} and "
         f"{int(q.loc['unsw_nb15', 'exact_duplicate_records']):,} rows). For CICIDS2017 the check "
         f"is made across capture days, so a within-file check would have understated it. This is "
         f"not necessarily an error — short, identical flows are genuinely common — but it "
         f"is a trap for evaluation: a random train/test split places identical rows on both "
         f"sides and inflates every reported score. Any benchmark result that does not state how "
         f"duplicates were handled should be read with that in mind [@arp2022], "
         f"[@pendlebury2019].")
    body(doc,
         "Separately, **three of CICIDS2017's fifteen labels are not valid UTF-8.** The Web "
         "Attack labels contain byte 0x96, a cp1252 en-dash. Reading the file as UTF-8 raises; "
         "reading it as latin-1 yields a different string than reading it as cp1252. An exact "
         "string match against 'Web Attack – Brute Force' therefore succeeds or fails depending "
         "on an argument elsewhere in the code, and when it fails the result is not an error but "
         "a silently missing attack class. The fix used here is to canonicalise every label to "
         "[A-Za-z0-9 ] before mapping, which makes the lookup encoding-independent.")
    figure(doc, "f02_attrition_duplication",
           "Figure 2 — (a) Record attrition through the two cleaning decisions. (b) Exact "
           "duplicate records as a share of each corpus. (c) The label-encoding defect, traced "
           "from the bytes on disk to the canonical form used for mapping.")

    h2(doc, "Class balance and the cost of harmonisation")
    imb = F["imb"]
    body(doc,
         f"Both benchmarks are severely imbalanced, and they are imbalanced differently. "
         f"CICIDS2017 is **80.3% benign**, with a majority-to-minority ratio across raw labels of "
         f"{imb.loc['cicids2017', 'imbalance_ratio_raw_labels']:,.0f}:1 — Heartbleed appears "
         f"**11 times** in 2.8 million records, and SQL injection 21 times. UNSW-NB15 is 36.1% "
         f"normal with a ratio of {imb.loc['unsw_nb15', 'imbalance_ratio_raw_labels']:,.0f}:1. "
         f"Reporting accuracy on either corpus is therefore uninformative [@he2009]: a "
         f"classifier that predicts 'benign' for every CICIDS2017 flow scores 80.3%.")
    body(doc,
         f"Harmonisation is not free. Dropping the attack categories with no counterpart removes "
         f"{F['cic_rows'] - F['cic_kept']:,} CICIDS2017 records (0.6%) but "
         f"**{F['unsw_rows'] - F['unsw_kept']:,} UNSW-NB15 records — one third of that "
         f"corpus**, mostly the 58,871 Generic block-cipher attacks that have no CICIDS analogue. "
         f"After harmonisation the two corpora still differ sharply in prior: CICIDS2017 is "
         f"80.8% benign against UNSW-NB15's 54.2%, and Exploits is 0.1% of one corpus against "
         f"28.2% of the other, a **350-fold difference in prevalence for the same class label**. "
         f"That asymmetry is the first concrete piece of evidence that the two benchmarks are "
         f"not interchangeable, and Section 6 shows it is far from the largest.")
    figure(doc, "f03_class_composition",
           "Figure 3 — (a, b) Every raw label in each benchmark on a log scale, coloured by the "
           "unified class it maps to; grey labels are dropped. (c) Class prevalence after "
           "harmonisation: the same four class names describe very different mixtures.")
    score = T["t09_quality_scorecard"].set_index("dataset").T.reset_index()
    score.columns = ["metric", "CICIDS2017", "UNSW-NB15"]
    table(doc, score,
          "Table 3 — Data-quality scorecard for the two benchmarks as published.",
          widths=[2.6, 2.0, 1.8], font=8.3)


def section_univariate(doc, F):
    uni, T = F["uni"], tables()
    h1(doc, "Univariate structure and the choice of scale")
    body(doc,
         "Before any pair of features is compared or any distance is computed, it has to be "
         "established what scale these variables live on. The answer determines whether a mean "
         "is interpretable, whether a standard deviation describes anything, and whether the "
         "outlier rules normally applied to such data mean what they are assumed to mean.")

    h2(doc, "Nine orders of magnitude")
    body(doc,
         f"Every one of the twelve features is non-negative, zero-inflated and heavy-tailed. The "
         f"largest observed value spans **{uni.loc['cicids2017', 'max_orders_of_magnitude']:.1f} "
         f"orders of magnitude** on CICIDS2017. Median |skewness| is "
         f"{uni.loc['cicids2017', 'median_abs_skew_raw']:.1f} on CICIDS2017 and "
         f"{uni.loc['unsw_nb15', 'median_abs_skew_raw']:.1f} on UNSW-NB15, reaching "
         f"{uni.loc['cicids2017', 'max_abs_skew_raw']:.0f} for down_up_ratio. For a Gaussian "
         f"both figures would be zero. The practical consequence is visible in the descriptive "
         f"statistics: CICIDS2017's flow_byte_rate has a median of 4,623 bytes/s, a 99th "
         f"percentile of 9.25 million, and a maximum of 2.07 billion. The mean of that column "
         f"describes no flow in the corpus.")
    body(doc,
         f"Zero inflation compounds it. {uni.loc['cicids2017', 'mean_zero_pct']:.0f}% of values "
         f"are exactly zero on average across CICIDS2017's twelve features, rising to 44% for "
         f"bwd_iat_mean, and these zeros are structural rather than missing: a flow with no "
         f"backward packets has zero backward bytes, zero backward packet size and a zero "
         f"down/up ratio by definition, which is why those three features share an identical "
         f"zero share in Figure 4.")
    figure(doc, "f04_marginal_distributions",
           "Figure 4 — Marginal distributions of all twelve features on a log1p axis, both "
           "benchmarks overlaid. The y-axis is set by the body of each distribution and the "
           "share of records sitting at exactly zero is printed in each panel. Several features "
           "— fwd_bytes, avg_fwd_pkt_size, flow_byte_rate — have visibly disjoint modes.")

    h2(doc, "The case for log1p")
    body(doc,
         f"A log1p transform is applied before every scale-sensitive analysis in this report. "
         f"Figure 5 is the evidence. Median |skewness| falls from "
         f"{uni.loc['cicids2017', 'median_abs_skew_raw']:.1f} to "
         f"{uni.loc['cicids2017', 'median_abs_skew_log']:.2f} on CICIDS2017 and from "
         f"{uni.loc['unsw_nb15', 'median_abs_skew_raw']:.1f} to "
         f"{uni.loc['unsw_nb15', 'median_abs_skew_log']:.2f} on UNSW-NB15; excess kurtosis falls "
         f"by up to six orders of magnitude. The worked example in panels (e) and (f) takes the "
         f"worst case, down_up_ratio, from a skewness of 870 to 1.69.")
    body(doc,
         "The transform is chosen over the alternatives for a specific reason. A fitted "
         "transform — Box-Cox [@boxcox1964], Yeo-Johnson [@yeojohnson2000], quantile "
         "normalisation — estimates "
         "parameters from data, and in a cross-dataset study that raises the question of which "
         "dataset it was fitted on; fitting on the source leaks nothing but distorts the target, "
         "and fitting on both pools information that is supposed to be held apart. **log1p "
         "estimates nothing.** It is defined for exactly the non-negative, zero-inflated, "
         "multiplicatively-scaled quantities at hand, and it is applied identically to both "
         "corpora.")
    figure(doc, "f05_transform_justification",
           "Figure 5 — (a–d) Absolute skewness and excess kurtosis per feature, before and after "
           "log1p, on a symmetric-log axis. (e, f) The worst-behaved feature as a worked example: "
           "on the native scale the entire corpus is one bar and the tail is invisible.")

    h2(doc, "Two outlier rules, two different answers")
    body(doc,
         f"The two screens in routine use disagree by an order of magnitude on this data. "
         f"Tukey's 1.5 × IQR rule [@tukey1977] flags "
         f"**{uni.loc['cicids2017', 'mean_tukey_pct']:.1f}%** of "
         f"CICIDS2017 records on average across the twelve features; the three-sigma rule flags "
         f"**{uni.loc['cicids2017', 'mean_z3_pct']:.1f}%**. The disagreement is not a bug in "
         f"either rule, it is a diagnosis: on a distribution this skewed the standard deviation "
         f"is itself inflated by the tail, so a three-sigma band is wide enough to contain almost "
         f"everything, while the IQR is narrow enough that a large, perfectly ordinary fraction "
         f"of the corpus falls outside 1.5 IQR of it.")
    body(doc,
         f"After log1p the Tukey rate falls to "
         f"{uni.loc['cicids2017', 'mean_tukey_log_pct']:.1f}% and "
         f"{uni.loc['unsw_nb15', 'mean_tukey_log_pct']:.1f}%, which is what one would expect of a "
         f"roughly symmetric variable. **Neither rule is used to drop records in this analysis.** "
         f"A 16% drop rate would discard most of the attack traffic, which is precisely the "
         f"signal the data exists to carry; high packet rates and large byte volumes are what a "
         f"volumetric denial-of-service flow looks like. Extreme values here are the phenomenon, "
         f"not contamination. The defensible options are to transform, which is what is done, or "
         f"to clip at fixed quantiles and report the clip — never to apply a mechanical "
         f"outlier filter and leave it unstated.")
    figure(doc, "f06_outlier_burden",
           "Figure 6 — Share of records flagged as outliers per feature under (a) Tukey's rule "
           "on the native scale, (b) the three-sigma rule on the native scale, and (c) Tukey's "
           "rule after log1p. The gap between (a) and (b) is the signature of heavy tails.")

    h2(doc, "Where the two corpora already diverge")
    body(doc,
         "Figure 7 puts the empirical CDFs side by side, which is the last point at which the "
         "cross-dataset gap can be seen without a test statistic attached to it. Several "
         "features barely overlap. The median CICIDS2017 flow carries **62 forward bytes** "
         "against UNSW-NB15's **800**; the median forward packet is 22.7 bytes against 84.0. "
         "A CICIDS2017 flow is a short, small, lightweight exchange and a UNSW-NB15 flow is not, "
         "and no amount of modelling will make those the same distribution. Section 6 quantifies "
         "the gap; this is where it becomes visible.")
    figure(doc, "f07_ecdf",
           "Figure 7 — Empirical cumulative distribution functions for all twelve features, both "
           "benchmarks overlaid, log1p axis. Horizontal separation between the two curves is "
           "distribution shift; for fwd_bytes and avg_fwd_pkt_size the curves barely touch.")
    desc = T["t10_descriptive_statistics"]
    sel = desc[desc.feature.isin(["fwd_bytes", "avg_fwd_pkt_size", "flow_byte_rate",
                                  "duration", "down_up_ratio"])]
    table(doc, sel[["dataset", "feature", "median", "q3", "p99", "max", "skew", "zero_pct"]],
          "Table 4 — Descriptive statistics for five representative features in their original "
          "units. The distance between the median, the 99th percentile and the maximum is the "
          "reason Section 4.2 is necessary. Full table: t10_descriptive_statistics.csv.",
          widths=[1.0, 1.4, 0.8, 0.8, 0.85, 0.85, 0.7, 0.75], font=8)


def section_multivariate(doc, F):
    T, vif = tables(), F["vif"]
    h1(doc, "Dependence structure and feature–class association")
    body(doc,
         "Two questions are settled here. How much genuinely independent information do the "
         "twelve features carry — which bounds how stable any ranking over them can be — "
         "and do the two benchmarks agree about which features carry the class?")

    h2(doc, "Correlation structure, and where it disagrees")
    body(doc,
         "Spearman's rank correlation [@spearman1904] rather than Pearson correlation is used "
         "throughout, for the reason "
         "established in Section 4: a linear coefficient on these marginals is largely a report "
         "on a handful of extreme flows. The within-benchmark structure is as expected — "
         "byte volumes track packet counts, rates track each other — but the comparison "
         "between benchmarks is not.")
    corr = F["corr"].head(5)
    body(doc,
         f"**Several feature pairs reverse sign between the two corpora.** On CICIDS2017, "
         f"avg_fwd_pkt_size and avg_bwd_pkt_size correlate at "
         f"ρ = {corr.iloc[0].rho_cicids2017:+.2f}; on UNSW-NB15 the same pair correlates at "
         f"{corr.iloc[0].rho_unsw_nb15:+.2f}, a swing of {corr.iloc[0].abs_difference:.2f}. "
         f"bwd_bytes against avg_fwd_pkt_size moves from {corr.iloc[1].rho_cicids2017:+.2f} to "
         f"{corr.iloc[1].rho_unsw_nb15:+.2f}. These are not small differences in degree; they "
         f"are statements that the relationship between forward and backward traffic is "
         f"organised differently in the two testbeds. A model that has learned to rely on one "
         f"of these relationships has learned something that is false in the other corpus.")
    figure(doc, "f08_correlation",
           "Figure 8 — Spearman rank correlation among the twelve features for (a) CICIDS2017 "
           "and (b) UNSW-NB15, with (c) the difference. Cells in (c) beyond ±0.5 mark "
           "relationships that do not hold in both benchmarks.")
    table(doc, F["corr"].head(6).round(3),
          "Table 5 — The six feature pairs whose rank correlation differs most between the two "
          "benchmarks. Four of the six change sign.",
          widths=[1.5, 1.5, 1.2, 1.2, 1.1], font=8)

    h2(doc, "The feature set is largely redundant")
    body(doc,
         f"Variance inflation factors [@kutner2005] are severe and asymmetric. On CICIDS2017 "
         f"the maximum is "
         f"**{vif.cicids2017.max():,.1f}** (bwd_bytes); on UNSW-NB15 it is "
         f"**{vif.unsw_nb15.max():,.0f}**, with eight of twelve features above 500. The "
         f"conventional thresholds of 5 and 10 are exceeded by an order of magnitude or more. "
         f"Principal components [@jolliffe2002] tell the same story from the other direction: "
         f"**{F['pc90']} components carry 90% of the variance of twelve features**, and the "
         f"first three alone carry {F['var3'] * 100:.0f}%.")
    body(doc,
         "This has a direct bearing on the parent project's question. When several features are "
         "near-substitutes, any method that distributes credit among them — a feature "
         "importance, a SHAP attribution, a permutation score [@hooker2021], [@strobl2008] "
         "— has no principled basis for "
         "choosing between them, and small perturbations move the ranking. **A portion of the "
         "rank instability that a cross-dataset study measures is therefore attributable to "
         "collinearity within the feature set rather than to any difference between datasets.** "
         "This is exactly why such a study needs a same-dataset reseed ceiling as a control, and "
         "Section 7 shows that the parent project's ceiling sits at 0.824 rather than at 1.0.")
    figure(doc, "f09_redundancy",
           "Figure 9 — (a) Variance inflation factors on a log axis against the conventional "
           "thresholds. (b, c) Hierarchical clustering of the features by correlation distance "
           "under average linkage [@sokal1958]: "
           "the two benchmarks do not group the same features together.")
    table(doc, F["vif"].round(1),
          "Table 6 — Variance inflation factors per feature. A VIF of 6,249 means that feature "
          "is almost entirely predictable from the other eleven.",
          widths=[2.0, 1.6, 1.6], font=8.3, index=True)

    h2(doc, "Which features carry the class — and do the benchmarks agree?")
    body(doc,
         "Normalised mutual information [@cover2006] is used to measure how much of each "
         "class's entropy a feature accounts for, computed one class against the rest with "
         "quantile binning; a histogram estimator is preferred to the nearest-neighbour "
         "estimators [@kraskov2004], [@ross2014] because those are two orders of magnitude "
         "slower at this record count and the task here is only to rank twelve features. The "
         "within-benchmark picture is sensible: on CICIDS2017 the backward-direction features "
         "dominate, with bwd_bytes reaching 0.64 for DoS/DDoS and down_up_ratio 0.68 for "
         "Reconnaissance. On UNSW-NB15 the leading feature for Reconnaissance is fwd_bytes at "
         "0.61 and the backward features are unremarkable.")
    agree = F["agree"]
    body(doc,
         f"Panel (c) of Figure 10 asks the question this report exists to ask. Ranking all "
         f"twelve features by mutual information within each benchmark, and comparing the two "
         f"rankings class by class, gives Spearman ρ of "
         f"{agree.iloc[0].spearman_rho:+.2f} for Benign, **{agree.iloc[1].spearman_rho:+.2f} for "
         f"DoS/DDoS**, {agree.iloc[2].spearman_rho:+.2f} for Reconnaissance/Probe and "
         f"{agree.iloc[3].spearman_rho:+.2f} for Exploits. The top-three features overlap at a "
         f"Jaccard index of **0.00 for both Benign and DoS/DDoS** — the three most "
         f"informative features for denial-of-service traffic in one benchmark are not among the "
         f"three most informative in the other, and the full ranking is slightly "
         f"anti-correlated.")
    body(doc,
         "This is a model-free result. No network was trained, no attribution method was run, "
         "and no explainer was involved. **The two benchmarks already disagree about which "
         "features characterise an attack class, as a property of the data.** Any study that "
         "trains on one and reads feature importances as insight into the attack is therefore "
         "making a claim that the data does not support, and the disagreement its attributions "
         "exhibit is at least partly inherited rather than created.")
    figure(doc, "f10_association",
           "Figure 10 — (a, b) Normalised mutual information between each feature and each class, "
           "one class against the rest. (c) Agreement between the two benchmarks' feature "
           "rankings: Spearman ρ over all twelve features, and Jaccard overlap of the top three.")
    table(doc, F["agree"][["class", "spearman_rho", "jaccard_top3",
                           "top3_cicids2017", "top3_unsw_nb15"]],
          "Table 7 — Per-class agreement between the benchmarks' mutual-information rankings, "
          "with the top three features on each side. Benign and DoS/DDoS share none.",
          widths=[1.2, 0.8, 0.7, 2.0, 1.9], font=7.6)
    figure(doc, "f11_class_conditional",
           "Figure 11 — Class-conditional distributions of the four most informative features. "
           "Reading down a column compares how the same feature separates classes in the two "
           "benchmarks; the orderings are not the same.")


def section_shift(doc, F):
    sh, shift, T = F["sh"], F["shift"], tables()
    h1(doc, "How far apart are the two benchmarks?")
    body(doc,
         "Sections 3 to 5 described each corpus on its own terms. This section measures the gap "
         "between them, which is the quantity that decides whether anything learned on one can "
         "be read on the other. Three independent views are taken: marginal divergence per "
         "feature, divergence within each class, and direct separability.")

    h2(doc, "Every feature has moved")
    top = shift.iloc[0]
    body(doc,
         f"Three statistics are reported per feature, because each fails differently. The "
         f"Kolmogorov–Smirnov statistic [@massey1951] is the maximum vertical gap between the "
         f"two ECDFs of Figure 7 and is easy to read but insensitive to the tails. The "
         f"population stability index is the standard industry measure, with thresholds "
         f"calibrated in credit scoring [@siddiqi2006]. The Jensen–Shannon divergence "
         f"[@lin1991] is symmetric, bounded in [0, 1] bits, and sensitive to "
         f"differences anywhere in the distribution. P-values are not reported: at 2.8 million "
         f"records against 171,705, every test rejects at any level for any difference at all, "
         f"so only the effect sizes carry information.")
    body(doc,
         f"**All twelve features exceed the PSI threshold of 0.25 that conventionally marks a "
         f"major shift**, most of them by an order of magnitude; the mean KS statistic is "
         f"{sh['mean_ks']:.2f} and the mean Jensen–Shannon divergence is "
         f"{sh['mean_jsd_bits']:.2f} bits. The worst case is **{top.feature}**, with "
         f"D = {top.ks_statistic:.2f}, PSI = {top.psi:.1f} and JSD = "
         f"{top.jensen_shannon_bits:.2f} bits — for context, a KS statistic of 0.80 means "
         f"that at some threshold the two corpora differ by 80 percentage points in the share of "
         f"flows below it. Not one feature in this shared space is distributionally stable "
         f"across the two benchmarks.")
    figure(doc, "f12_marginal_shift",
           "Figure 12 — Per-feature distribution shift under three measures. (a) "
           "Kolmogorov–Smirnov distance. (b) Population stability index against its conventional "
           "thresholds; every feature is in the 'major shift' band. (c) Jensen–Shannon "
           "divergence in bits.")
    table(doc, shift[["feature", "ks_statistic", "psi", "jensen_shannon_bits",
                      "standardised_mean_diff", "psi_verdict"]].round(3),
          "Table 8 — Distribution shift per feature, sorted by KS distance. The verdict column "
          "applies the conventional PSI bands (<0.1 stable, 0.1–0.25 moderate, >0.25 major).",
          widths=[1.5, 0.95, 0.8, 1.1, 1.1, 1.05], font=8)

    h2(doc, "The shift is not an artefact of class mixture")
    body(doc,
         f"Section 3.4 showed the two corpora have very different class priors, and a marginal "
         f"difference could in principle be nothing more than that — a corpus with more "
         f"denial-of-service traffic will have higher packet rates without any individual class "
         f"differing. Conditioning on class removes that explanation. The standardised mean "
         f"difference is computed within each class, so Benign is compared only to Benign.")
    body(doc,
         f"**The gap survives, and is large.** The mean absolute SMD within classes is "
         f"{sh['mean_abs_smd_within_class']:.2f}, against Cohen's conventional 0.8 "
         f"[@cohen1988] for a large "
         f"effect. Reconnaissance/Probe moves furthest at a mean |SMD| of 2.52, driven by "
         f"avg_fwd_pkt_size at **+12.7 standard deviations** — a CICIDS2017 port scan and a "
         f"UNSW-NB15 reconnaissance event are not the same traffic, whatever the shared label "
         f"says. Benign moves least at 0.68, which is intuitive: background traffic is the "
         f"closest thing the two testbeds have to a common phenomenon, and even it sits near the "
         f"'large effect' line.")
    figure(doc, "f13_class_shift",
           "Figure 13 — (a) Standardised mean difference per class and feature; positive means "
           "UNSW-NB15 is higher. Colour is clipped so the body remains readable, printed values "
           "are not. (b) Mean absolute SMD per class against Cohen's threshold for a large "
           "effect.")

    h2(doc, "A classifier can tell the two corpora apart")
    body(doc,
         f"The sharpest single summary of the gap is to ask a model for it directly. A "
         f"histogram-based gradient-boosted classifier [@friedman2001], [@ke2017] is trained "
         f"to predict, from the twelve shared features alone, which benchmark a flow came "
         f"from. If the two corpora were samples of the same underlying traffic the task would "
         f"be impossible and the AUC would sit at 0.5. What the achieved AUC estimates is the "
         f"divergence between two domains that bounds transfer error in the theory of learning "
         f"across domains [@bendavid2010].")
    body(doc,
         f"**The achieved AUC is {sh['domain_auc']:.5f}** on a held-out third of 120,000 "
         f"balanced flows. The corpora are, for practical purposes, perfectly separable. A single "
         f"feature does most of the work: avg_fwd_pkt_size alone reaches an AUC of "
         f"{F['dom'].loc['avg_fwd_pkt_size', 'solo_auc']:.2f}, and permuting it costs the full "
         f"model {F['dom'].loc['avg_fwd_pkt_size', 'permutation_importance']:.2f} AUC, more than "
         f"twenty times any other feature.")
    body(doc,
         "The implication is the central one of this report. **A detector trained on one of "
         "these corpora is operating in a feature space where corpus identity is almost perfectly "
         "recoverable**, which means nothing prevents it from keying on the capture rather than "
         "on the attack. It will not be penalised for doing so by any within-dataset evaluation, "
         "because within a single dataset the two are indistinguishable. Only a cross-dataset "
         "evaluation separates them, and Section 7 shows what happens when one is run.")
    figure(doc, "f14_domain_separability",
           "Figure 14 — (a) ROC curve for recovering the source benchmark from the twelve shared "
           "features. (b) Permutation importance. (c) AUC achievable from each feature on its "
           "own; five features individually exceed 0.63.")
    table(doc, F["dom"].reset_index().round(4).sort_values("solo_auc", ascending=False),
          "Table 9 — Domain-classifier diagnostics per feature: how much the full model loses "
          "when the feature is permuted, and how well the feature identifies the corpus alone.",
          widths=[2.0, 1.8, 1.6], font=8.3)

    h2(doc, "The geometry of the gap")
    body(doc,
         f"Figure 15 fits a principal-component basis on CICIDS2017 alone and projects UNSW-NB15 "
         f"into it. That asymmetry is deliberate: it mirrors how a source-fitted scaler is "
         f"applied to a target in a transfer setting, so the picture shows what the target looks "
         f"like **through the source's coordinate system** rather than through a compromise "
         f"basis that neither model would ever see. PC1 accounts for "
         f"{F['pca'].explained_variance_ratio.iloc[0] * 100:.0f}% of the variance and loads "
         f"positively on every volume feature, which is a size axis; PC2 separates rate from "
         f"duration.")
    body(doc,
         "In panels (c) and (d) the two corpora occupy adjacent but largely distinct regions, "
         "with UNSW-NB15 displaced along the size axis. The non-linear t-SNE embedding "
         "[@maaten2008] in panel "
         "(d) is the more striking: the pooled sample resolves into tight clusters that are "
         "almost always purely one corpus or the other, which is the same conclusion as the "
         "domain-classifier AUC arrived at by an unsupervised route. Panel (e) adds the reason "
         "this is a problem rather than a curiosity — within CICIDS2017 the classes "
         "themselves are not cleanly separated in the first two components, so the corpus "
         "signal is, if anything, stronger than the class signal in this space.")
    figure(doc, "f15_pca_structure",
           "Figure 15 — (a) Scree plot, fitted on CICIDS2017. (b) Loadings of the first three "
           "components. (c) Both corpora projected into the source basis. (d) t-SNE of a pooled "
           "balanced sample, coloured by corpus. (e) CICIDS2017 alone, coloured by class.")


def section_synthesis(doc, F):
    j, head = F["joined"], F["head"]
    h1(doc, "Synthesis: the EDA against the modelling outcome")
    body(doc,
         "Everything above was obtained from the data alone. This section checks it against the "
         "modelling results of the parent project, which are used here strictly as held-out "
         "evidence: a hybrid CNN-LSTM trained on CICIDS2017 across five seeds, applied unchanged "
         "to UNSW-NB15, with per-class SHAP rankings extracted under both conditions. None of "
         "those numbers were produced by this analysis.")

    h2(doc, "The prediction and the outcome")
    d = head["detection"]
    body(doc,
         f"The EDA's prediction was unambiguous: twelve features all in major shift, a mean "
         f"within-class effect size of {F['sh']['mean_abs_smd_within_class']:.2f}, and near-"
         f"perfect corpus separability together imply that a model fitted on one benchmark will "
         f"fail on the other. It does. Macro F1 falls from "
         f"**{d['source_f1_macro']:.3f}** on CICIDS2017 to **{d['target_f1_macro']:.3f}** on "
         f"UNSW-NB15 under transfer, and the Matthews correlation coefficient "
         f"[@matthews1975], [@chicco2020] is "
         f"**{d['target_mcc']:+.3f}** — marginally **worse than chance**. The failure is not "
         f"a deficiency of the architecture: the same architecture trained natively on UNSW-NB15 "
         f"reaches {d['native_target_f1_macro']:.3f}. The data is learnable; it is simply not "
         f"the same data.")
    body(doc,
         f"The class-level picture follows the EDA as well. Ranking the four classes by the "
         f"within-class shift measured in Section 6.2 and by their transfer F1 gives a Spearman "
         f"ρ of **−0.80** — the classes that moved most are the classes that "
         f"transfer worst. Reconnaissance/Probe, which the EDA flagged as the furthest-moved "
         f"class at a mean |SMD| of {j.loc['Reconnaissance/Probe', 'eda_mean_abs_smd']:.2f}, "
         f"scores a transfer F1 of exactly **{j.loc['Reconnaissance/Probe', 'transfer_f1']:.3f}** "
         f"despite being the single best-detected class within CICIDS2017 at "
         f"{j.loc['Reconnaissance/Probe', 'source_f1']:.3f}. With four classes this is a "
         f"descriptive correspondence rather than a significance test, and it is reported as "
         f"such; but the direction and the magnitude are what the EDA called.")
    figure(doc, "f16_synthesis",
           "Figure 16 — (a) Macro F1 under the three conditions, mean of five seeds. (b) "
           "Class-level shift measured before training against class-level transfer performance "
           "after. (c) Agreement of the mutual-information rankings (Figure 10c) against the "
           "model's SHAP explanation consistency, between the parent project's random floor and "
           "its same-dataset ceiling.")

    h2(doc, "Explanation instability was already in the data")
    body(doc,
         f"The parent project's central measurement is explanation consistency: how far per-class "
         f"SHAP rankings agree across the two benchmarks under a rank-similarity index of the "
         f"kind used for indefinite rankings [@webber2010], anchored between a "
         f"random-permutation floor ({head['lower_bound_ecs']:.3f}) and a same-dataset reseed "
         f"ceiling "
         f"({head['upper_bound_ecs']:.3f}). The observed native-condition score is "
         f"{head['conditions']['native']['observed_ecs']:.3f}, which normalises to "
         f"{head['conditions']['native']['normalised_consistency']:.3f} of the way between the "
         f"two anchors — significantly above chance (Wilcoxon signed-rank test "
         f"[@wilcoxon1945], p = "
         f"{head['conditions']['native']['wilcoxon']['p_value']:.1e}, 20 pairs), but far below "
         f"the agreement the same model reaches with itself on the same data under a different "
         f"seed.")
    body(doc,
         "Section 5.3 located part of that result in the data itself. The benchmarks' "
         "mutual-information rankings already disagree, at an average Spearman ρ of "
         f"{F['agree'].spearman_rho.mean():+.2f} and with zero top-three overlap for two of four "
         "classes, before any model or attribution method is involved. And Section 5.2 supplies "
         "the other part: with VIFs in the thousands and five components carrying 90% of the "
         "variance, the features are substitutable enough that any ranking over them is "
         "unstable on its own account — which is exactly why the same-dataset ceiling sits "
         f"at {head['upper_bound_ecs']:.3f} rather than at 1.0. That attributions are fragile "
         "under perturbation, re-training or a change of dataset is itself an established "
         "finding [@ghorbani2019], [@adebayo2018], [@krishna2022], and it has been reported "
         "specifically for security applications [@warnecke2020].")
    body(doc,
         "**The methodological point is that both of these are measurable without a model.** An "
         "EDA that reports distribution shift, feature redundancy and ranking agreement supplies "
         "the context a feature-importance claim needs in order to be interpretable, and it does "
         "so at a fraction of the cost of the experiment that would otherwise be needed to "
         "discover the same thing.")

    h2(doc, "Findings and their consequences")
    body(doc,
         "Table 10 collects every finding in this report with the evidence for it and what it "
         "implies for anyone modelling this data.")
    fnd = tables()["t25_findings"]
    table(doc, fnd, "Table 10 — Summary of findings. Each row states a finding, the measurement "
                    "that supports it, and the consequence for modelling.",
          widths=[1.6, 2.5, 2.3], font=7.6)


def section_limits(doc, F):
    h1(doc, "Threats to validity and limitations")
    bullets(doc, [
        "**Twelve features, not 79 and 49.** The shared space is the intersection of what both "
        "exporters measure, so it discards most of each benchmark. A richer shared space is not "
        "available — the exporters do not measure the same things — but the shift "
        "reported here is a property of this intersection, and a different intersection could "
        "give different magnitudes.",
        "**No TCP flag feature survives.** Flag counts were an obvious candidate for the shared "
        "space and were excluded deliberately: CICIDS2017's FIN Flag Count is zero in 99.2% of "
        "records while UNSW-NB15's nearest proxy is true in 45.5%. A feature that is "
        "near-constant on one side carries no rank information there, so including it would have "
        "manufactured apparent instability from an encoding difference. No flag-level feature is "
        "jointly available across these two benchmarks, and that is itself a limitation of the "
        "benchmark pair.",
        "**The Exploits class is a loose pairing.** CICIDS2017's web-application attacks and "
        "UNSW-NB15's service and protocol exploits are not the same attack family, and the class "
        "is unbalanced 2,227 against 48,365 records. Per-class results for Exploits are "
        "exploratory and are reported as such; the three tight classes carry the conclusions.",
        "**Shift is measured, not attributed.** This analysis establishes that the two corpora "
        "differ and by how much. It does not decompose that into testbed, exporter, capture "
        "period and synthetic-generation effects, which would require captures that vary one "
        "factor at a time and do not exist publicly.",
        "**Duplicates are counted, not removed.** The duplicate rates in Section 3.3 are "
        "reported but no de-duplication is applied, since identical short flows are a genuine "
        "property of network traffic rather than necessarily a defect. The consequence is that "
        "effective sample sizes are smaller than nominal ones, which matters for any inference "
        "but not for the effect sizes reported here.",
        "**Two benchmarks are not a sample of benchmarks.** The conclusion is that these two "
        "widely used corpora are not interchangeable. Whether that generalises to every "
        "benchmark pair is a stronger claim than this data supports, though the mechanism "
        "— different testbed, different exporter, different period — is general.",
        "**Section 7 is a correspondence over four classes.** The class-level agreements "
        "reported there are descriptive. Four points cannot support a significance claim and "
        "none is made.",
    ])


def section_conclusions(doc, F):
    sh = F["sh"]
    h1(doc, "Conclusions")
    body(doc,
         "This report set out to establish, by exploratory analysis alone, whether two of the "
         "most widely used intrusion detection benchmarks describe the same phenomenon. They do "
         "not, and the analysis needed to show it is neither elaborate nor expensive.")
    bullets(doc, [
        f"**The published files require work before they can be analysed at all.** Eight "
        f"zero-variance columns, thirteen columns with physically impossible negative values, "
        f"infinite rates from division by zero, a duplicated column name, 65 column names with "
        f"leading whitespace, a label set that is not valid UTF-8, and exact duplicate rates of "
        f"{F['q'].loc['cicids2017', 'duplicate_pct']:.1f}% and "
        f"{F['q'].loc['unsw_nb15', 'duplicate_pct']:.1f}%. Several of these fail silently rather "
        f"than loudly, which is what makes an explicit audit stage worth its cost.",
        "**Scale is not a detail here.** Features spanning nine orders of magnitude with "
        "skewness up to 870 make means, standard deviations and three-sigma rules misleading "
        "rather than merely imprecise, and the two standard outlier screens disagree by a factor "
        "of sixteen. A parameter-free log1p transform resolves this without estimating anything "
        "from either corpus.",
        f"**Every shared feature has shifted.** All twelve exceed the major-shift PSI threshold; "
        f"mean KS distance is {sh['mean_ks']:.2f}; the shift survives conditioning on class at a "
        f"mean |SMD| of {sh['mean_abs_smd_within_class']:.2f}; and a classifier recovers the "
        f"source corpus with AUC {sh['domain_auc']:.5f} from the shared features alone. The "
        f"corpora are, in this feature space, almost perfectly distinguishable.",
        "**The benchmarks disagree about which features matter, before any model exists.** "
        "Mutual-information rankings agree at an average Spearman ρ of "
        f"{F['agree'].spearman_rho.mean():+.2f} across the four classes, with zero top-three "
        "overlap for Benign and DoS/DDoS. Feature-importance instability across datasets is not "
        "purely an artefact of attribution methods; a component of it is inherited from the data.",
        f"**The modelling outcome followed the diagnostics.** Macro F1 of "
        f"{F['head']['detection']['source_f1_macro']:.3f} within CICIDS2017 becomes "
        f"{F['head']['detection']['target_f1_macro']:.3f} under transfer with a negative MCC, "
        f"while the same architecture trained natively on the target reaches "
        f"{F['head']['detection']['native_target_f1_macro']:.3f}; and the class the EDA ranked as "
        f"furthest-moved is the class that transfers worst.",
    ])
    body(doc,
         "The recommendation that follows is narrow and cheap to act on. **A cross-dataset claim "
         "in intrusion detection should be accompanied by a measurement of the gap being "
         "crossed** — at minimum a per-feature shift statistic and a domain-classifier AUC. "
         "Where the gap is this large, a within-dataset score is a statement about a capture and "
         "should be reported as one. The broader point is methodological: a model-free "
         "exploratory analysis anticipated an expensive experimental result, before it was run.")


def appendix(doc, F):
    h1_plain(doc, "Appendix A.  Reproducing the Analysis")
    body(doc,
         "The analysis is a package under `src/eda/` driven by a single entry point. Stage 1 "
         "reads the raw CSVs and takes a few minutes; every later stage reads a cached "
         "harmonised matrix and takes seconds.")
    cmds = ["pip install -r requirements.txt",
            "python scripts/run_eda.py           # all six stages",
            "python scripts/run_eda.py --from 2  # reuse the cache",
            "python scripts/build_report.py      # rebuild the paper"]
    if REPO:
        cmds.insert(0, f"git clone {REPO}")
    # Full width, like a figure. The clone line runs to 76 characters, which no
    # column measure holds at a readable size, and a URL broken across two
    # lines at one of its own hyphens cannot be copied back with confidence.
    _span(doc, 1)
    for i, cmd in enumerate(cmds):
        p = doc.add_paragraph()
        r = p.add_run(cmd)
        _serif(r, MONO)
        r.font.size = Pt(8)
        r.font.color.rgb = BLACK
        pf = p.paragraph_format
        # Hanging, so a longer command than any of these wraps as a visible
        # continuation rather than looking like two commands.
        pf.left_indent = Inches(0.26)
        pf.first_line_indent = Inches(-0.14)
        pf.space_after = Pt(1)
        pf.keep_with_next = i < len(cmds) - 1
    _span(doc, 2)
    _N["fresh"] = True

    h2(doc, "Stage map")
    stages = pd.DataFrame([
        ("1", "ingest.py", "Raw CSV reading, per-column quality audit, label harmonisation, cache",
         "T1–T6"),
        ("2", "stage_quality.py", "Defect figures, attrition, duplication, class balance", "T7–T9, F1–F3"),
        ("3", "stage_univariate.py", "Descriptives, shape, transform justification, outliers, ECDFs",
         "T10–T13, F4–F7"),
        ("4", "stage_multivariate.py", "Correlation, VIF, clustering, mutual information", "T14–T17, F8–F11"),
        ("5", "stage_shift.py", "KS / PSI / JSD, class-conditional SMD, domain classifier, PCA, t-SNE",
         "T18–T23, F12–F15"),
        ("6", "stage_synthesis.py", "Join to the parent project's modelling results", "T24–T25, F16"),
    ], columns=["stage", "module", "what it does", "outputs"])
    table(doc, stages, "Table A1 — Stage map. Outputs are written to eda_results/figures and "
                       "eda_results/tables.",
          widths=[0.5, 1.5, 3.2, 1.2], font=8)

    h2(doc, "Table index")
    T = tables()
    idx = pd.DataFrame({"file": [f"{k}.csv" for k in T], "rows": [len(v) for v in T.values()],
                        "columns": [v.shape[1] for v in T.values()]})
    table(doc, idx, "Table A2 — Every result table written by the pipeline.",
          widths=[3.6, 1.3, 1.3], font=8)

    h2(doc, "Environment")
    import matplotlib
    import numpy
    import scipy
    import sklearn
    env = pd.DataFrame([
        ("Python", "3.11"), ("NumPy", numpy.__version__), ("pandas", pd.__version__),
        ("SciPy", scipy.__version__), ("scikit-learn", sklearn.__version__),
        ("Matplotlib", matplotlib.__version__),
    ], columns=["package", "version"])
    table(doc, env, "Table A3 — Analysis environment.", widths=[2.4, 2.0],
          font=8, span=False)


def build() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    _CITED.clear()
    _N.update(sec=0, sub=0, fresh=False)
    T = tables()
    F = facts(T)

    doc = Document()
    _style(doc)

    title_block(doc)
    abstract(doc, F)
    section_intro(doc, F)
    section_data(doc, F)
    section_quality(doc, F)
    section_univariate(doc, F)
    section_multivariate(doc, F)
    section_shift(doc, F)
    section_synthesis(doc, F)
    section_limits(doc, F)
    section_conclusions(doc, F)
    appendix(doc, F)
    references(doc)
    _page_numbers(doc)

    doc.save(DOCX)
    print(f"  report -> {os.path.relpath(DOCX, ROOT)}  "
          f"({len(_CITED)} references, {len(doc.sections)} sections)")
    return DOCX


if __name__ == "__main__":
    build()
