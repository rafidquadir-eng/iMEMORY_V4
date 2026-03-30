#!/usr/bin/env python3
"""
step13_study_population_characterization.py
iMEMORY V4 — Study Population Demographics & Clinical Metadata

PURPOSE:
  Generates the clinical study characterization figures required for
  Nature Medicine / high-impact journal submissions:

  1. CONSORT/PRISMA-style Study Population Flow Diagram
     Shows total GEO records screened → eligible → included per disease class,
     broken down by modality, with exclusion reasons at each step.

  2. Table 1: Study Population Characteristics
     Per disease class AND per named subcluster:
       - N samples, N unique GEO series, N unique platforms/studies
       - Modality breakdown (bulk RNA-seq, ATAC-seq, methylation, scRNA, etc.)
       - Organism (human vs mouse)
       - Tissue type (PBMC, whole blood, kidney biopsy, monocytes, etc.)
       - Study design (cross-sectional, longitudinal, case-control)
       - Publication year range
       - Known clinical context from GEO title parsing

  3. Study Distribution Map Figure
     Bubble chart: x=publication year, y=disease category, bubble size=N samples,
     color=modality. Shows temporal distribution of included studies.

  4. Per-Subcluster Clinical Context Heatmap
     Shows which clinical phenotypes are represented in each of the 25 named
     subclusters (from GEO title keyword extraction).

METHOD:
  1. Parse accession_id column from endotype CSV to extract GSE series IDs
  2. Query NCBI GEO eutils API (esummary) for each unique GSE:
       https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi
       ?db=gds&id=<GSE_numeric_id>&retmode=json
  3. Parse returned JSON for: title, organism, type, overall_design,
     platform, pubmed_uid, gpl_name, n_samples, submission_date
  4. Categorize by tissue type and study design from title/description text
  5. Build summary statistics per cluster and overall
  6. Generate all figures + Table 1 CSV

OUTPUTS (all → /mnt/user-data/outputs/):
  study_manifest.csv              — one row per unique GSE study
  table1_overall.csv              — Table 1 overall demographics
  table1_per_cluster.csv          — Table 1 per named subcluster
  Figure_StudyFlow.png            — CONSORT/PRISMA flow diagram
  Figure_StudyDistribution.png    — bubble chart year × disease × modality
  Figure_SubclusterContext.png    — clinical context per subcluster

NOTE: NCBI eutils rate limit = 10 requests/second (3/sec without API key).
Script uses 0.34s delay between requests. Total runtime ~20-30 min for
1,478 unique accessions → ~200-300 unique GSE series.
"""
import os, sys, re, time, json, urllib.request, urllib.parse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter
import pathlib, random
random.seed(42)

ENDO_PATH = '/mnt/nvme15tb/trained_im/outputs/V4/stage3/endotype_assignments_v4_final.csv'
GRAPH_DIR = '/mnt/nvme15tb/trained_im/graphs_lmdb_v4'
OUT_ROOT  = '/mnt/nvme15tb/trained_im/outputs/V4/profiling/demographics/'
COPY_DIR  = '/mnt/user-data/outputs/'
pathlib.Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
# COPY_DIR disabled

NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_ESEARCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
DELAY_SEC     = 0.35   # stay under 3 req/sec (no API key)

# Named cluster definitions (from step2 log + step3)
NAMED_CLUSTERS = {
    242: {'name':'CKD_Glomerulonephritis',   'disease':'CKD',     'dc':[2]},
    354: {'name':'CKD_Diabetic_nephropathy', 'disease':'CKD',     'dc':[2]},
     80: {'name':'CKD_Glomerular_disease',   'disease':'CKD',     'dc':[2]},
    350: {'name':'CKD_IgA_nephropathy',      'disease':'CKD',     'dc':[2]},
    267: {'name':'CKD_ADPKD',               'disease':'CKD',     'dc':[2]},
    313: {'name':'CKD_Focal_segmental_GS',  'disease':'CKD',     'dc':[2]},
     12: {'name':'CKD_Peripheral_blood',    'disease':'CKD',     'dc':[2]},
    429: {'name':'HD_Uremia_dialysis',       'disease':'HD',      'dc':[3]},
    407: {'name':'HD_Innate_immune_act',     'disease':'HD',      'dc':[3]},
    262: {'name':'HD_ESRD_early',            'disease':'HD',      'dc':[3]},
    344: {'name':'HD_ESRD_late',             'disease':'HD',      'dc':[3]},
    337: {'name':'HD_PBMC',                  'disease':'HD',      'dc':[3]},
    394: {'name':'HD_Pre_post_dialysis',     'disease':'HD',      'dc':[3]},
    395: {'name':'HD_ESRD_periph_blood',     'disease':'HD',      'dc':[3]},
    236: {'name':'KTx_Acute_rej_Tcell',      'disease':'KTx',     'dc':[0]},
    452: {'name':'KTx_Acute_rej_AbMed',      'disease':'KTx',     'dc':[0]},
    509: {'name':'KTx_Acute_rej_mixed',      'disease':'KTx',     'dc':[0]},
    264: {'name':'KTx_Chronic_fibrosis',     'disease':'KTx',     'dc':[1]},
    104: {'name':'KTx_Chronic_IFTA',         'disease':'KTx',     'dc':[1]},
    417: {'name':'KTx_Chronic_allograft',    'disease':'KTx',     'dc':[1]},
    387: {'name':'KTx_CNI_inhibitor',        'disease':'KTx',     'dc':[4]},
    361: {'name':'KTx_Long_term_stable',     'disease':'KTx',     'dc':[4]},
    347: {'name':'KTx_CNI_minimization',     'disease':'KTx',     'dc':[4]},
    455: {'name':'KTx_Graft_monitoring',     'disease':'KTx',     'dc':[4]},
    464: {'name':'KTx_Multi_center_biopsy',  'disease':'KTx',     'dc':[4]},
}

DISEASE_COLORS = {'CKD':'#C62828','HD':'#1565C0','KTx':'#2E7D32','Control/TI':'#6A1B9A'}
MODALITY_COLORS = {
    'RNA-seq/gene expression': '#FF6B35',
    'ATAC-seq/chromatin':      '#4CC9F0',
    'Methylation/RRBS':        '#C2185B',
    'scRNA-seq':               '#9B5DE5',
    'miRNA':                   '#2EC4B6',
    'SNP/genotyping':          '#FFD60A',
    'Metabolomics':            '#A8DADC',
    'Multi-omics':             '#FF8A65',
    'Other/unknown':           '#AAAAAA',
}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.15})


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Parse endotype CSV and extract study manifest
# ══════════════════════════════════════════════════════════════════════════════
print("=== STEP 1: Parse endotype CSV ===")
endo = pd.read_csv(ENDO_PATH)
print(f"Total samples: {len(endo):,}")

# Map disease class
CLASS_NAMES = {0:'KTx_AcuteRej', 1:'KTx_ChronicRej', 2:'CKD', 3:'HD',
               4:'KTx_Stable', 5:'Control_TI'}
endo['disease_name'] = endo['disease_class'].map(CLASS_NAMES)
endo['main_disease'] = endo['disease_class'].map(
    {0:'KTx',1:'KTx',2:'CKD',3:'HD',4:'KTx',5:'Control/TI'})

# Map named cluster
endo['cluster_name'] = endo['refined_endotype'].map(
    {cid: m['name'] for cid, m in NAMED_CLUSTERS.items()}).fillna('unlabeled')

# Extract GSE series ID from accession_id
def extract_gse(acc_id):
    """Extract GEO series prefix from accession_id like 'GSE249006_s001234'."""
    if not isinstance(acc_id, str): return None
    m = re.match(r'^(GSE\d+|GSM\d+|E-[A-Z]+-\d+|ST\d+|ERP\d+|SRP\d+)', acc_id)
    return m.group(1) if m else acc_id.split('_')[0]

endo['gse_id'] = endo['accession_id'].apply(extract_gse)

# Count samples per GSE
gse_counts = endo.groupby('gse_id').agg(
    n_samples        = ('accession_id','count'),
    disease_classes  = ('disease_class', lambda x: sorted(set(x.astype(int)))),
    main_diseases    = ('main_disease', lambda x: list(set(x))),
    cluster_names    = ('cluster_name', lambda x: list(set(x) - {'unlabeled'})),
).reset_index()

print(f"Unique GSE series: {len(gse_counts):,}")
print(f"Samples per disease class:")
for dc, n in endo.groupby('disease_class')['accession_id'].count().items():
    print(f"  {CLASS_NAMES[dc]}: {n:,}")

# Detect modality from LMDB (scan one graph per accession)
print("\nDetecting modalities from LMDB graphs (sampling 1 per GSE)...")
import lmdb, pickle
lmdb_idx = {}
for fn in os.listdir(GRAPH_DIR):
    fp = os.path.join(GRAPH_DIR, fn)
    if fn.endswith('.lmdb') and os.path.isdir(fp):
        lmdb_idx[fn.replace('.lmdb','')] = fp

def detect_modality(node_types):
    nts = set(node_types)
    if 'gene_expression' in nts and len(nts) <= 3: return 'RNA-seq/gene expression'
    if 'gene_expression' in nts and any('scrna' in str(n).lower() for n in nts): return 'scRNA-seq'
    if 'chromatin_region' in nts: return 'ATAC-seq/chromatin'
    if 'cpg' in nts: return 'Methylation/RRBS'
    if 'mirna' in nts: return 'miRNA'
    if 'snp' in nts: return 'SNP/genotyping'
    if 'metabolite' in nts: return 'Metabolomics'
    if 'gene_expression' in nts: return 'RNA-seq/gene expression'
    return 'Other/unknown'

gse_modality = {}
gse_node_types = {}
for gse in gse_counts['gse_id'].tolist():
    # Sample first available key for this GSE
    sub = endo[endo['gse_id']==gse]
    for key in sub['accession_id'].tolist()[:10]:
        acc = key.rsplit('_s',1)[0]
        if acc not in lmdb_idx: continue
        try:
            env = lmdb.open(lmdb_idx[acc], readonly=True, lock=False)
            with env.begin() as txn:
                for k,v in txn.cursor():
                    g = pickle.loads(v)
                    nts = list(g.node_types)
                    gse_modality[gse]   = detect_modality(nts)
                    gse_node_types[gse] = nts
                    break
            env.close()
            if gse in gse_modality: break
        except: pass

gse_counts['modality']   = gse_counts['gse_id'].map(gse_modality).fillna('Other/unknown')
gse_counts['node_types'] = gse_counts['gse_id'].map(gse_node_types)

print(f"\nModality distribution:")
for mod, n in Counter(gse_counts['modality']).most_common():
    print(f"  {mod}: {n} studies")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Fetch GEO metadata via NCBI eutils
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 2: Fetch GEO metadata from NCBI ===")

def ncbi_esummary(gse_id, retries=3):
    """Fetch GEO series summary from NCBI eutils. Returns dict or None."""
    # Convert GSE12345 → numeric 12345
    m = re.match(r'GSE(\d+)', gse_id)
    if not m: return None
    numeric_id = m.group(1)

    url = (f"{NCBI_ESUMMARY}?db=gds&id={numeric_id}&retmode=json")
    for attempt in range(retries):
        try:
            time.sleep(DELAY_SEC)
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            result_dict = data.get('result', {})
            if not result_dict: return None
            # GDS UID is returned as key in result dict
            uids = result_dict.get('uids', [])
            if not uids: return None
            return result_dict.get(str(uids[0]), None)
        except Exception as e:
            if attempt < retries-1:
                time.sleep(2)
            else:
                return None
    return None


def parse_tissue(title, summary=''):
    """Extract tissue/cell type from GEO title."""
    text = (title + ' ' + summary).lower()
    tissue_map = [
        ('pbmc',            'PBMC'),
        ('peripheral blood mononuclear', 'PBMC'),
        ('monocyte',        'Monocytes'),
        ('macrophage',      'Macrophages'),
        ('whole blood',     'Whole blood'),
        ('kidney biopsy',   'Kidney biopsy'),
        ('renal biopsy',    'Kidney biopsy'),
        ('glomerulus',      'Glomerulus'),
        ('tubular',         'Kidney tubular'),
        ('urinary',         'Urine/urinary'),
        ('bone marrow',     'Bone marrow'),
        ('cd14',            'CD14+ monocytes'),
        ('cd4',             'CD4+ T cells'),
        ('t cell',          'T cells'),
        ('b cell',          'B cells'),
        ('nk cell',         'NK cells'),
        ('mait',            'MAIT cells'),
        ('dendritic',       'Dendritic cells'),
        ('spleen',          'Spleen'),
        ('lymph node',      'Lymph node'),
        ('serum',           'Serum'),
        ('plasma',          'Plasma'),
        ('urine',           'Urine'),
    ]
    for keyword, label in tissue_map:
        if keyword in text:
            return label
    return 'Not specified'

def parse_study_design(title, summary=''):
    """Classify study design from GEO title."""
    text = (title + ' ' + summary).lower()
    if any(w in text for w in ['time course','longitudinal','serial','follow-up','time-series']):
        return 'Longitudinal'
    if any(w in text for w in ['case-control','case control','versus healthy','vs healthy','compared to healthy']):
        return 'Case-control'
    if any(w in text for w in ['cohort','prospective','retrospective']):
        return 'Cohort'
    if any(w in text for w in ['cross-sectional','cross sectional','prevalence']):
        return 'Cross-sectional'
    if any(w in text for w in ['rct','randomized','clinical trial']):
        return 'Clinical trial'
    return 'Cross-sectional'  # default assumption

def parse_organism(organism_str):
    if not organism_str: return 'Unknown'
    org = str(organism_str).lower()
    if 'sapiens' in org or 'human' in org: return 'Human'
    if 'musculus' in org or 'mouse' in org: return 'Mouse'
    if 'rattus' in org or 'rat' in org: return 'Rat'
    return str(organism_str)


# Fetch metadata for all GSE series
geo_meta = {}
gse_list = [g for g in gse_counts['gse_id'].tolist() if str(g).startswith('GSE')]
print(f"Fetching NCBI metadata for {len(gse_list)} GSE series...")
print("(This will take ~{:.0f} min at 3 req/sec)".format(len(gse_list)*DELAY_SEC/60))

for i, gse in enumerate(gse_list):
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(gse_list)} ({100*i/len(gse_list):.0f}%)")
    meta = ncbi_esummary(gse)
    if meta:
        title    = meta.get('title', '')
        summary  = meta.get('summary', '')
        geo_meta[gse] = {
            'title':         title,
            'organism':      parse_organism(meta.get('taxon','')),
            'gpl':           meta.get('gpl',''),
            'n_samples_geo': meta.get('n_samples', 0),
            'submission_date': meta.get('submissiondate',''),
            'pub_date':      meta.get('pdat',''),
            'pubmed_id':     meta.get('pubmedids',''),
            'tissue':        parse_tissue(title, summary),
            'study_design':  parse_study_design(title, summary),
            'gse_type':      meta.get('gdstype',''),
            'summary':       summary[:200],
        }
    else:
        geo_meta[gse] = {
            'title':'GEO metadata unavailable', 'organism':'Unknown',
            'gpl':'', 'n_samples_geo':0, 'submission_date':'',
            'pub_date':'', 'pubmed_id':'', 'tissue':'Not specified',
            'study_design':'Cross-sectional', 'gse_type':'', 'summary':'',
        }

print(f"  Metadata fetched for {sum(1 for v in geo_meta.values() if v['title'] != 'GEO metadata unavailable')} / {len(gse_list)} series")

# Merge metadata into gse_counts
for col in ['title','organism','tissue','study_design','pub_date','pubmed_id','n_samples_geo','gse_type']:
    gse_counts[col] = gse_counts['gse_id'].map(
        lambda g: geo_meta.get(g, {}).get(col, 'Unknown') if str(g).startswith('GSE') else 'Non-GEO')

# Extract year
def extract_year(date_str):
    if not isinstance(date_str, str): return None
    m = re.search(r'(20\d{2}|19\d{2})', str(date_str))
    return int(m.group(1)) if m else None

gse_counts['year'] = gse_counts['pub_date'].apply(extract_year)
gse_counts.to_csv(os.path.join(OUT_ROOT,'study_manifest.csv'), index=False)
print(f"Saved study_manifest.csv ({len(gse_counts)} studies)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Build Table 1 (overall + per-cluster)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 3: Build Table 1 ===")

def build_table1(df_samples, gse_meta_df, group_col, group_vals, group_labels):
    """Build Table 1 rows for a set of groups."""
    rows = []
    for val, label in zip(group_vals, group_labels):
        if isinstance(val, list):
            mask = df_samples[group_col].isin(val)
        else:
            mask = df_samples[group_col] == val
        sub_s    = df_samples[mask]
        sub_gse  = gse_meta_df[gse_meta_df['gse_id'].isin(sub_s['gse_id'].unique())]

        n_samples   = len(sub_s)
        n_studies   = sub_s['gse_id'].nunique()
        years       = sub_gse['year'].dropna().astype(int)
        year_range  = f"{years.min()}–{years.max()}" if len(years)>0 else 'N/A'
        organisms   = sub_gse['organism'].value_counts()
        human_n     = organisms.get('Human', 0)
        mouse_n     = organisms.get('Mouse', 0)
        modalities  = sub_gse['modality'].value_counts()
        top_mod     = modalities.index[0] if len(modalities)>0 else 'Unknown'
        tissues     = sub_gse['tissue'].value_counts().head(3).index.tolist()
        designs     = sub_gse['study_design'].value_counts()
        longit_n    = designs.get('Longitudinal',0)
        pubmed_any  = sub_gse['pubmed_id'].apply(lambda x: bool(x) and x != '').sum()

        rows.append({
            'Group':                     label,
            'N samples':                 f"{n_samples:,}",
            'N GEO studies':             f"{n_studies}",
            'Year range':                year_range,
            'Human studies (n)':         f"{human_n}",
            'Mouse studies (n)':         f"{mouse_n}",
            'Primary modality':          top_mod,
            'Modality breakdown':        ', '.join(f"{k}:{v}" for k,v in modalities.items()),
            'Top tissues/cell types':    ' | '.join(tissues),
            'Longitudinal studies (n)':  f"{longit_n}",
            'With publication (n)':      f"{pubmed_any}",
        })
    return pd.DataFrame(rows)

# Merge gse_id into endo
endo_with_gse = endo.copy()

# Overall Table 1 by disease class
t1_overall = build_table1(
    endo_with_gse, gse_counts,
    group_col   = 'main_disease',
    group_vals  = ['CKD','HD','KTx','Control/TI'],
    group_labels= ['CKD (class 2)','Hemodialysis (class 3)',
                   'KTx / Rejection (classes 0,1,4)','Control / TI (class 5)']
)
t1_overall.to_csv(os.path.join(OUT_ROOT,'table1_overall.csv'), index=False)
print("Saved table1_overall.csv")
print(t1_overall[['Group','N samples','N GEO studies','Year range','Primary modality']].to_string(index=False))

# Per-subcluster Table 1
cluster_rows = []
for cid, meta in NAMED_CLUSTERS.items():
    sub_s   = endo_with_gse[(endo_with_gse['refined_endotype']==cid) &
                             endo_with_gse['disease_class'].isin(meta['dc'])]
    sub_gse = gse_counts[gse_counts['gse_id'].isin(sub_s['gse_id'].unique())]
    n_samp  = len(sub_s)
    n_stud  = sub_s['gse_id'].nunique()
    years   = sub_gse['year'].dropna().astype(int)
    yr_range = f"{years.min()}–{years.max()}" if len(years)>0 else 'N/A'
    mods    = sub_gse['modality'].value_counts()
    tissues = sub_gse['tissue'].value_counts().head(2).index.tolist()
    organisms = sub_gse['organism'].value_counts()
    designs = sub_gse['study_design'].value_counts()
    titles  = sub_gse['title'].dropna().tolist()

    cluster_rows.append({
        'Cluster ID':               cid,
        'Cluster name':             meta['name'],
        'Disease':                  meta['disease'],
        'N samples':                n_samp,
        'N GEO studies':            n_stud,
        'Year range':               yr_range,
        'Human (n studies)':        organisms.get('Human',0),
        'Mouse (n studies)':        organisms.get('Mouse',0),
        'Primary modality':         mods.index[0] if len(mods)>0 else 'Unknown',
        'Modalities':               ', '.join(f"{k}:{v}" for k,v in mods.items()),
        'Cell/tissue types':        ' | '.join(tissues),
        'Study designs':            ', '.join(f"{k}:{v}" for k,v in designs.items()),
        'Longitudinal (n)':         designs.get('Longitudinal',0),
        'With PubMed ID (n)':       sub_gse['pubmed_id'].apply(lambda x: bool(x) and x!='').sum(),
        'Example GEO titles':       ' // '.join(titles[:3]),
    })

t1_cluster = pd.DataFrame(cluster_rows)
t1_cluster.to_csv(os.path.join(OUT_ROOT,'table1_per_cluster.csv'), index=False)
print(f"Saved table1_per_cluster.csv ({len(t1_cluster)} clusters)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: CONSORT/PRISMA Study Population Flow Figure
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 4: CONSORT flow diagram ===")

# Compute exclusion stats
total_lmdb_shards     = len(lmdb_idx)
total_gse_identified  = len(gse_list) + (len(gse_counts) - len(gse_list))  # GSE + non-GSE
total_gse_queried     = len(gse_list)
gse_no_meta           = sum(1 for v in geo_meta.values() if v['title']=='GEO metadata unavailable')
gse_with_meta         = total_gse_queried - gse_no_meta

# Build counts per disease
disease_study_counts = {}
disease_sample_counts = {}
disease_modality_counts = {}
for disease in ['CKD','HD','KTx','Control/TI']:
    sub_s = endo_with_gse[endo_with_gse['main_disease']==disease]
    sub_g = gse_counts[gse_counts['gse_id'].isin(sub_s['gse_id'].unique())]
    disease_study_counts[disease]    = len(sub_g)
    disease_sample_counts[disease]   = len(sub_s)
    disease_modality_counts[disease] = dict(sub_g['modality'].value_counts().head(4))

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('white')
ax = fig.add_axes([0,0,1,1])
ax.set_xlim(0,100); ax.set_ylim(0,100)
ax.axis('off')

# ── Helper functions for CONSORT boxes ────────────────────────────────────────
def flow_box(ax, x, y, w, h, title, lines, color='#E3F2FD', title_color='#1565C0',
             fontsize=8.5, title_size=9.5):
    rect = patches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle='round,pad=0.5', facecolor=color,
                               edgecolor='#333333', lw=1.2, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y+h/2-1.5, title, ha='center', va='top', fontsize=title_size,
            fontweight='bold', color=title_color, zorder=3)
    for i, line in enumerate(lines):
        ax.text(x, y+h/2-3.5-i*2.0, line, ha='center', va='top',
                fontsize=fontsize, color='#333333', zorder=3)

def flow_arrow(ax, x1, y1, x2, y2, label='', label_side='right'):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5), zorder=1)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ha = 'left' if label_side=='right' else 'right'
        dx = 1.0 if label_side=='right' else -1.0
        ax.text(mx+dx, my, label, ha=ha, va='center', fontsize=7.5,
                color='#666666', style='italic')

def excl_box(ax, x, y, w, h, lines, color='#FFF3E0'):
    rect = patches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle='round,pad=0.4', facecolor=color,
                               edgecolor='#E65100', lw=1.0, linestyle='--', zorder=2)
    ax.add_patch(rect)
    ax.text(x, y+h/2-1.2, '⚠ Excluded:', ha='center', va='top', fontsize=7.5,
            fontweight='bold', color='#E65100', zorder=3)
    for i, line in enumerate(lines):
        ax.text(x, y+h/2-3.0-i*1.8, line, ha='center', va='top',
                fontsize=7.0, color='#666', zorder=3)

# Title
ax.text(50, 97, 'iMEMORY V4 — Study Population Flow Diagram',
        ha='center', va='top', fontsize=14, fontweight='bold', color='#111111')
ax.text(50, 94.5,
        f'Multi-modal GEO cohort | {len(endo):,} samples across {len(gse_counts):,} unique studies',
        ha='center', va='top', fontsize=10, color='#555555')

# Level 1: Total LMDB records
flow_box(ax, 50, 87, 40, 9, 'TOTAL RECORDS IDENTIFIED',
         [f'{total_lmdb_shards:,} LMDB graph databases scanned',
          f'{len(endo):,} sample records loaded',
          f'{len(gse_counts):,} unique accession series identified'],
         color='#E8EAF6', title_color='#283593')

# Arrow down
flow_arrow(ax, 50, 82.5, 50, 78.5)

# Level 2: GEO queried + metadata fetch
flow_box(ax, 50, 75, 40, 7, 'GEO METADATA QUERY (NCBI eutils)',
         [f'{total_gse_queried} GSE series queried via NCBI esummary',
          f'{gse_with_meta} series with complete metadata ({100*gse_with_meta//max(1,total_gse_queried)}%)',
          f'{gse_no_meta} series — metadata unavailable (included with LMDB data only)'],
         color='#E8F5E9', title_color='#2E7D32')

# Exclusion box right
excl_box(ax, 83, 75, 28, 7,
         [f'{gse_no_meta} series: GEO metadata not retrievable',
          '(non-human organisms may apply)',
          'Samples retained in embedding pipeline'])

flow_arrow(ax, 70, 75, 77, 75, label='', label_side='right')

# Arrow down
flow_arrow(ax, 50, 71.5, 50, 67.5)

# Level 3: Disease class assignment
flow_box(ax, 50, 64, 40, 7, 'DISEASE CLASS ASSIGNMENT & ELIGIBILITY',
         [f'{len(endo):,} samples assigned to 6 disease classes',
          f'Via V4 GNN encoder (macro F1=0.714, TI recall=0.925)',
          f'620 HDBSCAN clusters → 25 named subclusters'],
         color='#F3E5F5', title_color='#6A1B9A')

# Arrow down to 4 branches
flow_arrow(ax, 50, 60.5, 50, 57)

# Branch points (4 disease groups)
disease_x = [14, 36, 64, 86]
disease_y = 47.5
disease_h = 22
disease_groups = [
    ('CKD',        disease_study_counts.get('CKD',0),
     disease_sample_counts.get('CKD',0),   DISEASE_COLORS['CKD'],   '#FFEBEE'),
    ('HD',         disease_study_counts.get('HD',0),
     disease_sample_counts.get('HD',0),    DISEASE_COLORS['HD'],    '#E3F2FD'),
    ('KTx',        disease_study_counts.get('KTx',0),
     disease_sample_counts.get('KTx',0),   DISEASE_COLORS['KTx'],   '#E8F5E9'),
    ('Control/TI', disease_study_counts.get('Control/TI',0),
     disease_sample_counts.get('Control/TI',0), DISEASE_COLORS['Control/TI'], '#F3E5F5'),
]
disease_labels = ['Chronic Kidney\nDisease (CKD)','Hemodialysis\n(HD)',
                  'Kidney Transplant\n(KTx/Rejection)','Control /\nTrained Immunity']

# Horizontal connector line
ax.plot([14,86],[57,57], color='#333333', lw=1.5, zorder=1)
for dx in disease_x:
    ax.annotate('', xy=(dx, disease_y+disease_h/2), xytext=(dx, 57),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5), zorder=1)

for i, (dx, (disease, n_stud, n_samp, dc, bg), dlabel) in \
        enumerate(zip(disease_x, disease_groups, disease_labels)):
    mods = disease_modality_counts.get(disease, {})
    mod_str = [f"  {k.split('/')[0]}: {v}" for k,v in list(mods.items())[:3]]
    flow_box(ax, dx, disease_y, 20, disease_h, dlabel,
             [f'Studies: {n_stud}',
              f'Samples: {n_samp:,}'] + mod_str,
             color=bg, title_color=dc, fontsize=7.5, title_size=8.5)

# Arrow down to subclusters box
flow_arrow(ax, 50, disease_y-disease_h/2, 50, 13)

# Level 5: Named subclusters
cluster_per_disease = Counter(m['disease'] for m in NAMED_CLUSTERS.values())
flow_box(ax, 50, 10, 60, 9, 'NAMED SUBCLUSTER ASSIGNMENT (Step 2 log)',
         [f'7 CKD subclusters (n={disease_sample_counts.get("CKD",0):,} samples)',
          f'7 HD subclusters (n={disease_sample_counts.get("HD",0):,} samples)',
          f'11 KTx subclusters  (n={disease_sample_counts.get("KTx",0):,} samples)',
          f'Control / TI reference (n={disease_sample_counts.get("Control/TI",0):,} samples)',
          f'→ 620 HDBSCAN clusters total | 25 clinically named'],
         color='#FFF9C4', title_color='#F57F17', fontsize=8, title_size=9.5)

# Footnote
ax.text(50, 2.5,
        'iMEMORY V4 | GEO data downloaded Jan–Mar 2026 | '
        'Samples = individual LMDB graph records | '
        'Studies = unique GEO series (GSE) identifiers | '
        'Modalities: bulk RNA-seq, ATAC-seq, methylation (RRBS/EPIC), scRNA-seq, miRNA, SNP, metabolomics',
        ha='center', va='bottom', fontsize=6.5, color='#888888', style='italic')

p_flow = os.path.join(OUT_ROOT,'Figure_StudyFlow.png')
fig.savefig(p_flow, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
import shutil; # shutil.copy(p_flow, os.path.join(COPY_DIR,'Figure_StudyFlow.png'))
print(f"Saved: Figure_StudyFlow.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Study distribution bubble chart (year × disease × modality)
# ══════════════════════════════════════════════════════════════════════════════
print("=== STEP 5: Study distribution bubble chart ===")

# Merge disease info into gse_counts
endo_gse_map = endo_with_gse.groupby('gse_id')['main_disease'].first().reset_index()
gse_plot = gse_counts.merge(endo_gse_map, on='gse_id', how='left')
gse_plot = gse_plot.dropna(subset=['year'])
gse_plot['year'] = gse_plot['year'].astype(int)

# Also get n_samples from endo
n_samp_map = endo_with_gse.groupby('gse_id')['accession_id'].count().rename('n_samples_endo')
gse_plot   = gse_plot.merge(n_samp_map, on='gse_id', how='left')

# Y-axis jitter by disease
disease_y_map = {'CKD':1, 'HD':2, 'KTx':3, 'Control/TI':4}
np.random.seed(42)
gse_plot['y_pos'] = gse_plot['main_disease'].map(disease_y_map) + \
                    np.random.uniform(-0.3, 0.3, len(gse_plot))

fig2, ax2 = plt.subplots(figsize=(14, 7))
ax2.set_facecolor('#F8F9FA')
ax2.grid(axis='x', color='white', lw=1.0)
ax2.set_axisbelow(True)

for mod, color in MODALITY_COLORS.items():
    mask = gse_plot['modality'] == mod
    sub  = gse_plot[mask]
    if len(sub) == 0: continue
    sizes = np.sqrt(sub['n_samples_endo'].fillna(10)) * 6
    sizes = sizes.clip(20, 400)
    ax2.scatter(sub['year'], sub['y_pos'], s=sizes, c=color, alpha=0.7,
                linewidths=0.5, edgecolors='white', label=mod, zorder=3)

ax2.set_yticks([1,2,3,4])
ax2.set_yticklabels(['CKD','Hemodialysis (HD)','KTx / Rejection','Control / TI'],
                     fontsize=11, fontweight='bold')
ax2.set_xlabel('Year of GEO Submission', fontsize=11)
ax2.set_xlim(2003, 2026)
ax2.xaxis.set_major_locator(plt.MultipleLocator(2))

# Disease group shading
for y, color in [(1,'#FFEBEE'),(2,'#E3F2FD'),(3,'#E8F5E9'),(4,'#F3E5F5')]:
    ax2.axhspan(y-0.45, y+0.45, alpha=0.25, color=color, zorder=0)

ax2.legend(loc='upper left', fontsize=8, framealpha=0.9,
           title='Modality', title_fontsize=8.5,
           markerscale=0.8, handlelength=0.8)

# Bubble size legend
for sz, label in [(20,'~10 samples'),(100,'~300 samples'),(300,'~2,500 samples')]:
    ax2.scatter([], [], s=sz, c='grey', alpha=0.5, label=label)
ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.9,
           title='Bubble size', title_fontsize=8, markerscale=1.0)

ax2.set_title('iMEMORY V4 — Study Distribution by Year, Disease Class, and Modality\n'
              f'({len(gse_plot):,} GEO studies | {len(endo):,} samples | 2004–2025)',
              fontsize=12, fontweight='bold', pad=10)
for sp in ax2.spines.values(): sp.set_edgecolor('#CCC')

plt.tight_layout()
p_dist = os.path.join(OUT_ROOT,'Figure_StudyDistribution.png')
fig2.savefig(p_dist, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
import shutil; # shutil.copy(p_dist, os.path.join(COPY_DIR,'Figure_StudyDistribution.png'))
print(f"Saved: Figure_StudyDistribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Per-subcluster clinical context heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("=== STEP 6: Subcluster clinical context heatmap ===")

# Build context matrix: rows=subclusters, cols=key clinical features
CLINICAL_FEATURES = [
    ('Human', lambda r: r['Human (n studies)'] > 0),
    ('Mouse', lambda r: r['Mouse (n studies)'] > 0),
    ('Longitudinal', lambda r: r['Longitudinal (n)'] > 0),
    ('Published', lambda r: r['With PubMed ID (n)'] > 0),
    ('PBMC/monocytes', lambda r: any(t in str(r['Cell/tissue types'])
                                      for t in ['PBMC','Monocyte','monocyte','CD14'])),
    ('Kidney biopsy', lambda r: 'biopsy' in str(r['Cell/tissue types']).lower() or
                                 'kidney' in str(r['Cell/tissue types']).lower()),
    ('Whole blood', lambda r: 'blood' in str(r['Cell/tissue types']).lower()),
    ('Multi-modal', lambda r: len(str(r['Modalities']).split(',')) > 1),
    ('RNA-seq', lambda r: 'RNA' in str(r['Primary modality'])),
    ('ATAC-seq', lambda r: 'ATAC' in str(r['Modalities'])),
    ('Methylation', lambda r: 'Methyl' in str(r['Modalities'])),
    ('scRNA-seq', lambda r: 'scRNA' in str(r['Modalities'])),
]
feature_names = [f[0] for f in CLINICAL_FEATURES]

context_mat = np.zeros((len(t1_cluster), len(CLINICAL_FEATURES)))
for i, row in t1_cluster.iterrows():
    for j, (fname, func) in enumerate(CLINICAL_FEATURES):
        try:
            context_mat[i, j] = 1.0 if func(row) else 0.0
        except: pass

fig3, ax3 = plt.subplots(figsize=(14, max(6, len(t1_cluster)*0.42)))
im3 = ax3.imshow(context_mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1,
                  interpolation='nearest')

ax3.set_xticks(range(len(feature_names)))
ax3.set_xticklabels(feature_names, rotation=35, ha='right', fontsize=9)
ax3.set_yticks(range(len(t1_cluster)))

# Color the y-axis labels by disease
ylabels = [f"{row['Cluster name'].replace('CKD_','').replace('HD_','').replace('KTx_','')}\n"
           f"(n={row['N samples']:,}, {row['N GEO studies']}st)"
           for _, row in t1_cluster.iterrows()]
ax3.set_yticklabels(ylabels, fontsize=7.5)

# Color labels by disease
for i, (_, row) in enumerate(t1_cluster.iterrows()):
    col = DISEASE_COLORS.get(row['Disease'], '#333333')
    ax3.get_yticklabels()[i].set_color(col)

# Checkmark annotations
for i in range(len(t1_cluster)):
    for j in range(len(feature_names)):
        if context_mat[i,j] > 0:
            ax3.text(j, i, '✓', ha='center', va='center',
                     fontsize=9, color='white', fontweight='bold')

# Disease group separator lines
prev_disease = None
for i, (_, row) in enumerate(t1_cluster.iterrows()):
    if row['Disease'] != prev_disease and i > 0:
        ax3.axhline(i-0.5, color='white', lw=2.5)
    prev_disease = row['Disease']

ax3.set_title('iMEMORY V4 — Clinical Characteristics per Disease Subcluster\n'
              '(✓ = feature present in ≥1 study | color = disease group)',
              fontsize=11.5, fontweight='bold', pad=10)

handles = [mpatches.Patch(facecolor=c, label=d)
           for d,c in DISEASE_COLORS.items()]
ax3.legend(handles=handles, loc='lower right', fontsize=9,
           framealpha=0.92, bbox_to_anchor=(1.0,-0.05))

plt.tight_layout()
p_ctx = os.path.join(OUT_ROOT,'Figure_SubclusterContext.png')
fig3.savefig(p_ctx, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig3)
import shutil; # shutil.copy(p_ctx, os.path.join(COPY_DIR,'Figure_SubclusterContext.png'))
print(f"Saved: Figure_SubclusterContext.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Print summary statistics for Table 1 text
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 7: Summary statistics ===")
print("\n=== TABLE 1 OVERALL ===")
print(t1_overall.to_string(index=False))

print("\n=== TABLE 1 PER CLUSTER (abbreviated) ===")
print(t1_cluster[['Cluster name','Disease','N samples','N GEO studies',
                   'Year range','Human (n studies)','Primary modality',
                   'Cell/tissue types']].to_string(index=False))

# Overall statistics for paper text
total_human = gse_counts[gse_counts['organism']=='Human']['gse_id'].nunique()
total_mouse  = gse_counts[gse_counts['organism']=='Mouse']['gse_id'].nunique()
total_longit = gse_counts[gse_counts['study_design']=='Longitudinal']['gse_id'].nunique()
year_min = gse_counts['year'].dropna().astype(int).min()
year_max = gse_counts['year'].dropna().astype(int).max()
modality_counts = gse_counts['modality'].value_counts()

print(f"\n=== PAPER TEXT STATISTICS ===")
print(f"Total samples: {len(endo):,}")
print(f"Unique GEO series: {len(gse_counts):,}")
print(f"Human studies: {total_human}")
print(f"Mouse studies: {total_mouse}")
print(f"Longitudinal studies: {total_longit}")
print(f"Year range: {year_min}–{year_max}")
print(f"\nModality breakdown:")
for mod, n in modality_counts.items():
    print(f"  {mod}: {n} studies ({100*n/len(gse_counts):.1f}%)")

# Save all outputs
import shutil
for fname in ['table1_overall.csv','table1_per_cluster.csv','study_manifest.csv']:
    src = os.path.join(OUT_ROOT, fname)
    if os.path.exists(src):
        # shutil.copy(src, os.path.join(COPY_DIR, fname))
        print(f"Copied {fname} to outputs")

print(f"\n=== step13 DONE ===")
print(f"Outputs: {OUT_ROOT}")
print(f"Figures copied to: {COPY_DIR}")
