#!/usr/bin/env python3
"""
NPZ Extractor v4 - Fixed CEL decompression and improved robustness
"""

import os, sys, json, gzip, tarfile, zipfile, tempfile, shutil
import logging, argparse, threading, time, subprocess, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from collections import defaultdict

import boto3
import numpy as np
import scipy.io, scipy.sparse

sys.path.insert(0, '/mnt/data/trained_im/scripts')
from pipeline_config import PATHS, LABEL_MAP

BUCKET          = 'trained-im-disease-harvest'
NPZ_OUT_DIR     = '/mnt/nvme15tb/trained_im/NPZ_fresh'
TMP_DIR         = '/mnt/nvme15tb/trained_im/tmp_work'
LOG_FILE        = '/mnt/nvme15tb/trained_im/logs/extractor_v4.log'
DONE_FILE       = '/mnt/data/trained_im/metadata/extracted_accessions.txt'
FAILED_FILE     = '/mnt/data/trained_im/metadata/failed_accessions_v4.jsonl'
MAX_WORKERS     = 96
MAX_GENES       = 60000
MAX_SAMPLES     = 5000
MIN_SAMPLES     = 2
ACCESSION_TIMEOUT = 1800

for d in [NPZ_OUT_DIR, TMP_DIR, os.path.dirname(LOG_FILE), os.path.dirname(DONE_FILE)]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)s] %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('extractor')

_tls = threading.local()
def s3():
    if not hasattr(_tls, 'c'):
        _tls.c = boto3.client('s3', region_name='us-east-1')
    return _tls.c

_done_lock = threading.Lock()
_failed_lock = threading.Lock()

_done = set()
if os.path.exists(DONE_FILE):
    with open(DONE_FILE) as f:
        _done = {l.strip() for l in f if l.strip()}

def build_work_queue(modality_filter=None, retry_failed=False):
    log.info("Scanning S3 for accessions...")
    client = boto3.client('s3', region_name='us-east-1')
    paginator = client.get_paginator('list_objects_v2')

    prefix_map = defaultdict(list)
    for page in paginator.paginate(Bucket=BUCKET, Prefix='raw/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            parts = key.split('/')
            if len(parts) < 3:
                continue
            modality = parts[1]
            if len(parts) >= 4:
                prefix = '/'.join(parts[:3])
            else:
                fname = parts[2]
                acc = fname.split('_')[0]
                prefix = f'raw/{modality}/{acc}'
            prefix_map[prefix].append(key)

    log.info(f"Found {len(prefix_map)} accession prefixes")

    failed_accs = set()
    if retry_failed and os.path.exists(FAILED_FILE):
        with open(FAILED_FILE) as f:
            for line in f:
                try:
                    failed_accs.add(json.loads(line)['accession'])
                except:
                    pass

    work_items = []
    for prefix, keys in prefix_map.items():
        parts = prefix.split('/')
        modality = parts[1] if len(parts) > 1 else 'unknown'
        accession = parts[2] if len(parts) > 2 else parts[-1]

        if modality_filter and modality != modality_filter:
            continue
        if accession in _done and accession not in failed_accs:
            continue

        out_path = os.path.join(NPZ_OUT_DIR, f'{accession}.npz')
        if os.path.exists(out_path) and accession not in failed_accs:
            mark_done(accession)
            continue

        sidecar_key = next((k for k in keys if k.endswith('_disease_class.json')), None)
        if sidecar_key is None:
            continue

        try:
            resp = client.get_object(Bucket=BUCKET, Key=sidecar_key)
            sidecar = json.loads(resp['Body'].read())
        except:
            continue

        disease_class = sidecar.get('disease_class', '')
        if disease_class not in LABEL_MAP:
            continue

        data_keys = [k for k in keys if not k.endswith('_disease_class.json')]
        if not data_keys:
            continue

        work_items.append({
            'accession': accession,
            'modality': modality,
            'disease_class': disease_class,
            'keys': data_keys,
        })

    log.info(f"Work queue: {len(work_items)} accessions")
    return work_items


def download_and_extract(keys, work_dir):
    """Download S3 keys, extract archives, decompress files."""
    for key in keys:
        fname = key.split('/')[-1]
        dst = os.path.join(work_dir, fname)
        try:
            s3().download_file(BUCKET, key, dst)
        except Exception as e:
            log.warning(f"Download failed {key}: {e}")

    # Multi-pass extraction
    for _ in range(3):
        changed = False
        for root, dirs, files in os.walk(work_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                fnl = fn.lower()
                try:
                    if fnl.endswith('.tar.gz') or fnl.endswith('.tgz') or fnl.endswith('.tar'):
                        with tarfile.open(fp) as tar:
                            tar.extractall(root, filter='data')
                        os.remove(fp)
                        changed = True
                    elif fnl.endswith('.zip'):
                        with zipfile.ZipFile(fp) as z:
                            z.extractall(root)
                        os.remove(fp)
                        changed = True
                except Exception as e:
                    log.debug(f"Archive error {fp}: {e}")
        if not changed:
            break

    # Decompress CEL.gz, GPR.gz, RCC.gz files
    for root, dirs, files in os.walk(work_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            fnl = fn.lower()
            if fnl.endswith('.cel.gz') or fnl.endswith('.gpr.gz') or fnl.endswith('.rcc.gz'):
                try:
                    out = fp[:-3]
                    with gzip.open(fp, 'rb') as fi, open(out, 'wb') as fo:
                        shutil.copyfileobj(fi, fo)
                    os.remove(fp)
                    log.debug(f"Decompressed {fn}")
                except Exception as e:
                    log.warning(f"Decompress error {fp}: {e}")

    all_files = []
    for root, dirs, files in os.walk(work_dir):
        for fn in files:
            all_files.append(os.path.join(root, fn))
    return all_files


def ext_of(path):
    name = os.path.basename(path).lower()
    for ext in ['.fastq.gz','.fq.gz','.txt.gz','.csv.gz','.tsv.gz','.bed.gz',
                '.mtx.gz','.wig.gz','.cel.gz','.narrowpeak.gz','.broadpeak.gz',
                '.h5ad','.h5','.bw','.bigwig','.cel','.gpr','.rcc','.fcs','.xlsx','.xls']:
        if name.endswith(ext):
            return ext
    return Path(path).suffix.lower()


def parse_series_matrix(path):
    """Parse GEO series_matrix.txt.gz files."""
    import pandas as pd
    try:
        opener = gzip.open if path.endswith('.gz') else open
        data_lines = []
        in_table = False
        with opener(path, 'rt', errors='replace') as f:
            for line in f:
                if 'series_matrix_table_begin' in line.lower():
                    in_table = True
                    continue
                if 'series_matrix_table_end' in line.lower():
                    break
                if in_table:
                    data_lines.append(line)
        if len(data_lines) < 10:
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(''.join(data_lines)), sep='\t', index_col=0,
                         na_values=['', 'NA', 'null', 'NaN', 'nan'])
        df.index = df.index.astype(str).str.strip('"')
        df.columns = [str(c).strip('"') for c in df.columns]
        df = df.select_dtypes(include=[np.number])
        if df.empty or df.shape[0] < 10 or df.shape[1] < MIN_SAMPLES:
            return None
        return df.values.astype(np.float32), list(df.index), list(df.columns)
    except Exception as e:
        log.debug(f"series_matrix failed {path}: {e}")
        return None


def parse_count_matrix(path):
    """Generic delimited matrix parser."""
    import pandas as pd
    try:
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rt', errors='replace') as f:
            first_lines = [f.readline() for _ in range(10)]
        sep = '\t' if any('\t' in l for l in first_lines) else ','
        skip = sum(1 for l in first_lines if l.startswith('#') or l.startswith('!'))
        df = pd.read_csv(path, sep=sep, index_col=0, skiprows=skip, low_memory=False)
        df = df.select_dtypes(include=[np.number])
        if df.empty or df.shape[0] < 10 or df.shape[1] < MIN_SAMPLES:
            return None
        if df.shape[0] < df.shape[1]:
            df = df.T
        return df.values.astype(np.float32), list(df.index), list(df.columns)
    except Exception as e:
        log.debug(f"count_matrix failed {path}: {e}")
        return None


def parse_cel_r(cel_files, work_dir):
    """Parse CEL files using R/affy."""
    if not cel_files or len(cel_files) < MIN_SAMPLES:
        return None
    r_script = os.path.join(work_dir, 'cel_parse.R')
    out_csv = os.path.join(work_dir, 'cel_expr.csv')
    cel_list = ','.join([f'"{f}"' for f in cel_files])
    r_code = f'''
suppressPackageStartupMessages(library(affy))
cel_files <- c({cel_list})
setwd("{work_dir}")
tryCatch({{
    raw <- ReadAffy(filenames=cel_files)
    eset <- rma(raw)
    expr <- exprs(eset)
    write.csv(expr, "{out_csv}")
    cat("SUCCESS\\n")
}}, error=function(e) {{
    cat("FAILED:", conditionMessage(e), "\\n")
}})
'''
    with open(r_script, 'w') as f:
        f.write(r_code)
    try:
        result = subprocess.run(['Rscript', r_script], capture_output=True, 
                                text=True, timeout=600, cwd=work_dir)
        if 'SUCCESS' in result.stdout and os.path.exists(out_csv):
            import pandas as pd
            df = pd.read_csv(out_csv, index_col=0)
            if df.shape[0] >= 10 and df.shape[1] >= MIN_SAMPLES:
                return df.values.astype(np.float32), list(df.index), list(df.columns)
    except Exception as e:
        log.debug(f"CEL R failed: {e}")
    return None


def parse_h5ad(path):
    """Parse AnnData h5ad files."""
    try:
        import anndata
        adata = anndata.read_h5ad(path)
        if scipy.sparse.issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = np.array(adata.X)
        genes = list(adata.var_names) if len(adata.var_names) > 0 else [f'gene_{i}' for i in range(X.shape[1])]
        samples = list(adata.obs_names) if len(adata.obs_names) > 0 else [f'cell_{i}' for i in range(X.shape[0])]
        X = X.T
        if X.shape[0] < 10 or X.shape[1] < MIN_SAMPLES:
            return None
        return X.astype(np.float32), genes, samples
    except Exception as e:
        log.debug(f"h5ad failed {path}: {e}")
        return None


def parse_10x_mex(work_dir):
    """Parse 10x MEX format."""
    try:
        mtx_file = barcodes_file = features_file = None
        for root, dirs, files in os.walk(work_dir):
            for f in files:
                fl = f.lower()
                fp = os.path.join(root, f)
                if 'matrix.mtx' in fl:
                    mtx_file = fp
                elif 'barcodes' in fl:
                    barcodes_file = fp
                elif 'features' in fl or 'genes' in fl:
                    features_file = fp
        if mtx_file is None:
            return None
        if mtx_file.endswith('.gz'):
            with gzip.open(mtx_file, 'rb') as f:
                mat = scipy.io.mmread(f)
        else:
            mat = scipy.io.mmread(mtx_file)
        mat = mat.tocsr()
        if barcodes_file:
            opener = gzip.open if barcodes_file.endswith('.gz') else open
            with opener(barcodes_file, 'rt') as f:
                barcodes = [l.strip().split('\t')[0] for l in f]
        else:
            barcodes = [f'cell_{i}' for i in range(mat.shape[1])]
        if features_file:
            opener = gzip.open if features_file.endswith('.gz') else open
            with opener(features_file, 'rt') as f:
                features = [l.strip().split('\t')[0] for l in f]
        else:
            features = [f'gene_{i}' for i in range(mat.shape[0])]
        X = mat.toarray().astype(np.float32)
        if X.shape[0] < 10 or X.shape[1] < MIN_SAMPLES:
            return None
        return X, features, barcodes
    except Exception as e:
        log.debug(f"10x MEX failed: {e}")
        return None


def parse_bed_peaks(bed_files, work_dir):
    """Parse BED files into binary peak matrix."""
    if not bed_files or len(bed_files) < MIN_SAMPLES:
        return None
    all_peaks = []
    sample_names = []
    for bed_file in bed_files:
        try:
            opener = gzip.open if bed_file.endswith('.gz') else open
            peaks = set()
            with opener(bed_file, 'rt') as f:
                for line in f:
                    if line.startswith('#') or line.startswith('track'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        try:
                            chrom, start = parts[0], int(parts[1])
                            peaks.add(f"{chrom}:{start//1000*1000}")
                        except:
                            continue
            if peaks:
                all_peaks.append(peaks)
                sample_names.append(os.path.basename(bed_file).split('.')[0])
        except:
            continue
    if len(all_peaks) < MIN_SAMPLES:
        return None
    union_peaks = sorted(set.union(*all_peaks))
    if len(union_peaks) < 100:
        return None
    matrix = np.zeros((len(union_peaks), len(all_peaks)), dtype=np.float32)
    peak_to_idx = {p: i for i, p in enumerate(union_peaks)}
    for j, peaks in enumerate(all_peaks):
        for p in peaks:
            if p in peak_to_idx:
                matrix[peak_to_idx[p], j] = 1.0
    return matrix, union_peaks, sample_names


def parse_bigwig(bw_files, work_dir):
    """Parse BigWig files."""
    try:
        import pyBigWig
    except:
        return None
    if not bw_files or len(bw_files) < MIN_SAMPLES:
        return None
    regions = [(f'chr{i}', pos, pos+1000) for i in range(1, 23) for pos in range(0, 50000000, 10000)]
    all_data = {}
    for bw_file in bw_files:
        try:
            bw = pyBigWig.open(bw_file)
            sample_name = os.path.basename(bw_file).split('.')[0]
            values = []
            for chrom, start, end in regions:
                try:
                    val = bw.stats(chrom, start, end, type='mean')
                    values.append(val[0] if val[0] else 0.0)
                except:
                    values.append(0.0)
            all_data[sample_name] = np.array(values)
            bw.close()
        except:
            continue
    if len(all_data) < MIN_SAMPLES:
        return None
    region_names = [f"{c}:{s}" for c, s, e in regions]
    matrix = np.column_stack([all_data[s] for s in sorted(all_data.keys())])
    return matrix.astype(np.float32), region_names, sorted(all_data.keys())


def parse_excel(path):
    """Parse Excel files."""
    import pandas as pd
    try:
        df = pd.read_excel(path, index_col=0, engine='openpyxl')
        df = df.select_dtypes(include=[np.number])
        if df.empty or df.shape[0] < 10 or df.shape[1] < MIN_SAMPLES:
            return None
        if df.shape[0] < df.shape[1]:
            df = df.T
        return df.values.astype(np.float32), list(df.index), list(df.columns)
    except:
        return None


def mark_done(accession):
    with _done_lock:
        if accession not in _done:
            _done.add(accession)
            with open(DONE_FILE, 'a') as f:
                f.write(accession + '\n')


def record_failure(accession, modality, files):
    with _failed_lock:
        with open(FAILED_FILE, 'a') as f:
            f.write(json.dumps({
                'accession': accession,
                'modality': modality,
                'files': [os.path.basename(f) for f in files[:20]],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }) + '\n')


def process_accession(item, dry_run=False):
    accession = item['accession']
    modality = item['modality']
    disease_class = item['disease_class']
    keys = item['keys']
    out_path = os.path.join(NPZ_OUT_DIR, f'{accession}.npz')

    if os.path.exists(out_path):
        mark_done(accession)
        return out_path

    if dry_run:
        log.info(f"[DRY-RUN] {accession} ({modality}) {len(keys)} files")
        return None

    work_dir = os.path.join(TMP_DIR, accession)
    os.makedirs(work_dir, exist_ok=True)

    try:
        local_files = download_and_extract(keys, work_dir)
        if not local_files:
            log.warning(f"{accession}: No files after download/extract")
            return None

        files_by_ext = defaultdict(list)
        for f in local_files:
            files_by_ext[ext_of(f)].append(f)

        result = None
        parser_used = None

        # 1. h5ad
        h5ads = files_by_ext.get('.h5ad', [])
        if h5ads:
            result = parse_h5ad(h5ads[0])
            if result:
                parser_used = 'h5ad'

        # 2. 10x MEX
        if result is None and (files_by_ext.get('.mtx.gz') or files_by_ext.get('.mtx')):
            result = parse_10x_mex(work_dir)
            if result:
                parser_used = '10x_mex'

        # 3. Series matrix
        sm_files = [f for f in local_files if 'series_matrix' in f.lower()]
        if result is None and sm_files:
            for sm in sm_files:
                result = parse_series_matrix(sm)
                if result:
                    parser_used = 'series_matrix'
                    break

        # 4. CEL files
        cel_files = [f for f in local_files if f.lower().endswith('.cel')]
        if result is None and cel_files:
            log.debug(f"{accession}: Found {len(cel_files)} CEL files")
            result = parse_cel_r(cel_files, work_dir)
            if result:
                parser_used = 'CEL'

        # 5. BED/peaks
        bed_files = [f for f in local_files if any(f.lower().endswith(e) for e in 
                     ['.bed', '.bed.gz', '.narrowpeak', '.narrowpeak.gz', '.broadpeak', '.broadpeak.gz'])]
        if result is None and bed_files:
            result = parse_bed_peaks(bed_files, work_dir)
            if result:
                parser_used = 'BED'

        # 6. BigWig
        bw_files = [f for f in local_files if f.lower().endswith(('.bw', '.bigwig'))]
        if result is None and bw_files:
            result = parse_bigwig(bw_files, work_dir)
            if result:
                parser_used = 'BigWig'

        # 7. Excel
        excel_files = [f for f in local_files if f.lower().endswith(('.xlsx', '.xls'))]
        if result is None and excel_files:
            for ef in excel_files:
                result = parse_excel(ef)
                if result:
                    parser_used = 'Excel'
                    break

        # 8. Generic count matrix
        txt_files = [f for f in local_files if any(f.lower().endswith(e) for e in 
                     ['.txt', '.txt.gz', '.csv', '.csv.gz', '.tsv', '.tsv.gz'])
                     and 'series_matrix' not in f.lower()
                     and 'filelist' not in f.lower()]
        if result is None and txt_files:
            for tf in sorted(txt_files, key=lambda x: os.path.getsize(x), reverse=True):
                result = parse_count_matrix(tf)
                if result:
                    parser_used = 'count_matrix'
                    break

        if result is None:
            file_exts = list(files_by_ext.keys())
            log.warning(f"✗ {accession} ({modality}): No parser succeeded. Extensions: {file_exts}")
            record_failure(accession, modality, local_files)
            return None

        matrix, genes, samples = result
        if matrix is None or len(genes) == 0 or len(samples) == 0:
            log.warning(f"✗ {accession}: Empty result from {parser_used}")
            record_failure(accession, modality, local_files)
            return None

        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if matrix.shape[0] > MAX_GENES:
            var = np.var(matrix, axis=1)
            top_idx = np.argsort(var)[-MAX_GENES:]
            matrix = matrix[top_idx, :]
            genes = [genes[i] for i in top_idx]

        if matrix.shape[1] > MAX_SAMPLES:
            matrix = matrix[:, :MAX_SAMPLES]
            samples = samples[:MAX_SAMPLES]

        np.savez_compressed(out_path, matrix=matrix, genes=np.array(genes, dtype=object),
                            samples=np.array(samples, dtype=object), disease_class=disease_class,
                            modality=modality, accession=accession, parser=parser_used)

        log.info(f"✓ {accession} ({modality}): {matrix.shape[0]}x{matrix.shape[1]} via {parser_used}")
        mark_done(accession)
        return out_path

    except Exception as e:
        log.error(f"✗ {accession}: Exception: {e}")
        record_failure(accession, modality, local_files if 'local_files' in dir() else [])
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--modality', type=str, default=None)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"NPZ Extractor v4 - Workers: {args.workers}")
    log.info("=" * 60)

    work_queue = build_work_queue(args.modality, args.retry_failed)
    if not work_queue:
        log.info("No work to do!")
        return

    success = failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix='w') as executor:
        futures = {executor.submit(process_accession, item, args.dry_run): item for item in work_queue}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result(timeout=ACCESSION_TIMEOUT)
                if result:
                    success += 1
                else:
                    failed += 1
            except TimeoutError:
                log.error(f"✗ {item['accession']}: Timeout")
                failed += 1
            except Exception as e:
                log.error(f"✗ {item['accession']}: {e}")
                failed += 1
            total = success + failed
            if total % 50 == 0:
                elapsed = time.time() - start_time
                rate = total / elapsed * 3600 if elapsed > 0 else 0
                log.info(f"Progress: {total}/{len(work_queue)} ({success} success, {failed} failed) Rate: {rate:.0f}/hr")

    log.info("=" * 60)
    log.info(f"COMPLETE: {success} success, {failed} failed")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
