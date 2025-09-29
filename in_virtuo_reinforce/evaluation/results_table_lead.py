import re
import glob
import os
import pandas as pd
import numpy as np
import argparse
def extract_seed_metrics(results_dir):
    """
    Build {(target, idx): (seed_qed, seed_sa)} from any available thr file.
    Seed rows are detected as sim >= 0.999. If multiple, take the first.
    """
    rx = re.compile(r"docking_(?P<target>[\w\-]+)_idx(?P<idx>\d+)_thr(?P<thr>[46])\.csv")
    files = glob.glob(os.path.join(results_dir, "**/docking_*_idx*_thr*.csv"), recursive=True)

    metrics = {}
    seen = set()

    for fn in files:
        m = rx.search(os.path.basename(fn))
        if not m:
            continue
        target, idx = m.group("target"), int(m.group("idx"))
        key = (target, idx)
        if key in seen:
            continue

        try:
            df = pd.read_csv(fn)
            if all(col in df.columns for col in ["sim", "qed", "sa"]):
                seed_rows = df[df["sim"] >= 0.999]
                if not seed_rows.empty:
                    row0 = seed_rows.iloc[0]
                    seed_qed = float(row0["qed"])
                    seed_sa = float(row0["sa"])
                    metrics[key] = (seed_qed, seed_sa)
                    seen.add(key)
        except Exception as e:
            print(f"Warning: could not read seed metrics from {fn}: {e}")

    return metrics
def load_seed_metrics_from_actives(csv_path, baseline_raw_df, atol=0.05):
    """
    Map (target, idx) -> (QED, SA) using actives.csv.
    Rows are matched by DS ~= abs(seed_score) with tolerance.
    """
    if not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)
        df["SA"] = 10.0 -9* df["SA"]
    except Exception as e:
        print(f"Warning: could not read {csv_path}: {e}")
        return {}

    if not all(c in df.columns for c in ["target", "DS", "QED", "SA"]):
        print(f"Warning: missing required columns in {csv_path}")
        return {}

    per_target = {}
    for tgt, sub in df.groupby("target"):
        per_target[tgt] = [(float(r.DS), float(r.QED), float(r.SA)) for _, r in sub.iterrows()]

    metrics = {}
    for target in baseline_raw_df["target"].unique():
        target_df = baseline_raw_df[baseline_raw_df["target"] == target].reset_index(drop=True)
        candidates = per_target.get(target, [])
        for idx in range(len(target_df)):
            seed_score = float(target_df.iloc[idx]["seed_score"])
            want = abs(seed_score)
            hit = None
            for ds, q, s in candidates:
                if abs(ds - want) <= atol:
                    hit = (q, s)
                    break
            if hit:
                metrics[(target, idx)] = hit
    return metrics
def parse_baseline_table(raw_text):
    """
    Parse the baseline table from raw text into a structured DataFrame
    """
    lines = raw_text.strip().splitlines()

    rows = []
    current_target = None

    for line in lines:
        # Skip header line
        if 'Target' in line or line.strip() == '':
            continue

        # Split by | character
        parts = line.split('|')
        if len(parts) < 8:
            continue

        # Extract target if present
        if parts[0].strip() and not parts[0].strip().replace('.', '').replace('-', '').replace(' ', '').isdigit():
            current_target = parts[0].strip()

        # Skip if we don't have a current target
        if current_target is None:
            continue

        def parse_value(val):
            val = val.strip()
            if val == '-' or val == '':
                return None
            try:
                return float(val)
            except:
                return None

        # Parse the values
        try:
            seed_score = parse_value(parts[1])
            if seed_score is None:  # Skip if no seed score
                continue

            row = {
                'target': current_target,
                'seed_score': seed_score,
                'GenMol_0.4': parse_value(parts[2]),
                'RetMol_0.4': parse_value(parts[3]),
                'GraphGA_0.4': parse_value(parts[4]),
                'GenMol_0.6': parse_value(parts[5]),
                'RetMol_0.6': parse_value(parts[6]),
                'GraphGA_0.6': parse_value(parts[7]) if len(parts) > 7 else None,
            }
            rows.append(row)
        except:
            continue

    return pd.DataFrame(rows)

def baseline_df_to_multiindex(df):
    """
    Convert baseline DataFrame to MultiIndex format
    """
    # Group by target and assign indices
    targets = []
    indices = []
    data_dict = {}

    for target in df['target'].unique():
        target_df = df[df['target'] == target].reset_index(drop=True)
        for idx in range(len(target_df)):
            targets.append(target)
            indices.append(idx)

            row_data = target_df.iloc[idx]
            for threshold in ['0.4', '0.6']:
                for method in ['GenMol', 'RetMol', 'GraphGA']:
                    col_name = f'{method}_{threshold}'
                    value = row_data.get(col_name)

                    key = (target, idx, f'δ = {threshold}', method)
                    if pd.notna(value):
                        data_dict[key] = f"{value:.1f}"
                    else:
                        data_dict[key] = "-"

    # Create MultiIndex DataFrame
    row_index = pd.MultiIndex.from_arrays([targets, indices], names=['protein', 'idx'])

    # Create columns
    thresholds = ['δ = 0.4', 'δ = 0.6']
    methods = ['GenMol', 'RetMol', 'GraphGA']
    column_tuples = [(thr, method) for thr in thresholds for method in methods]
    columns = pd.MultiIndex.from_tuples(column_tuples, names=['threshold', 'method'])

    # Create DataFrame
    result_df = pd.DataFrame(index=row_index, columns=columns)

    # Populate DataFrame
    for (target, idx, threshold, method), value in data_dict.items():
        if (target, idx) in result_df.index and (threshold, method) in result_df.columns:
            result_df.loc[(target, idx), (threshold, method)] = value

    return result_df

# def load_your_results(results_dir, method_name="InVirtuoGen"):
#     """
#     Load your experimental results from CSV files
#     """
#     rx = re.compile(r"docking_(?P<target>[\w\-]+)_idx(?P<idx>\d+)_thr(?P<thr>[46])\.csv")
#     files = glob.glob(os.path.join(results_dir, "**/docking_*_idx*_thr*.csv"), recursive=True)

#     # Collect all unique values
#     targets = set()
#     idxs = set()
#     thrs = set()
#     thrs.add("0.")
#     for fn in files:
#         m = rx.search(os.path.basename(fn))
#         if not m:
#             continue
#         target, idx, thr = m.group("target"), int(m.group("idx")), m.group("thr")
#         targets.add(target)
#         idxs.add(idx)
#         thrs.add(thr)

#     targets = sorted(list(targets))
#     idxs = sorted(list(idxs))
#     thrs = sorted(list(thrs))

#     # Create results dictionary
#     results = {}

#     for target in targets:
#         results[target] = {}
#         for idx in idxs:
#             results[target][idx] = {}
#             for thr in thrs:
#                 fn = os.path.join(results_dir, f"docking_{target}_idx{idx}_thr{thr}.csv")
#                 if not os.path.exists(fn):
#                     results[target][idx][thr] = []
#                     continue

#                 try:
#                     df_temp = pd.read_csv(fn)

#                     # Check required columns
#                     required_cols = ['sim', 'qed', 'sa', 'seed', 'docking score']
#                     if not all(col in df_temp.columns for col in required_cols):
#                         print(f"Warning: Missing columns in {fn}")
#                         results[target][idx][thr] = []
#                         continue

#                     # Apply filters
#                     thr_val = float(thr) / 10.0
#                     df_filtered = df_temp[
#                         (df_temp['sim'] > thr_val) &
#                         (df_temp['qed'] > 0.6) &
#                         (df_temp['sa'] < 4)
#                     ]
#                     if df_filtered.empty:
#                         results[target][idx][thr] = []
#                         continue

#                     # Get best (max) docking score for each seed
#                     # Your scores are positive (absolute values), so max() gets the best score
#                     seed_best_scores = df_filtered.groupby("seed")["docking score"].max()
#                     # Convert to negative to match convention
#                     seed_best_scores = [-score for score in seed_best_scores.tolist()]
#                     results[target][idx][thr] = seed_best_scores

#                 except Exception as e:
#                     print(f"Error processing {fn}: {e}")
#                     results[target][idx][thr] = []
#     # Create MultiIndex DataFrame
#     row_tuples = [(target, idx) for target in targets for idx in idxs]
#     row_index = pd.MultiIndex.from_tuples(row_tuples, names=['protein', 'idx'])

#     column_tuples = [(f"δ = 0.{thr}", method_name) for thr in thrs]
#     columns = pd.MultiIndex.from_tuples(column_tuples, names=['threshold', 'method'])

#     df_final = pd.DataFrame(index=row_index, columns=columns)
#     # Populate the table
#     target_sums = {thr:0 for thr in thrs}
#     for target in targets:

#         for idx in idxs:
#             for thr in thrs:

#                 scores = results[target][idx][thr]

#                 if not scores:
#                     value = "-"
#                 elif len(scores) == 1:
#                     value = f"{scores[0]:.1f}"
#                 else:
#                     mean_val = np.mean(scores)
#                     std_val = np.std(scores, ddof=1)
#                     value = f"{mean_val:.1f}±{std_val:.1f}"
#                     target_sums[thr] += float(mean_val)
#                 df_final.loc[(target, idx), (f"δ = 0.{thr}", method_name)] = value
#     print("df_final", df_final)
#     print("target_sums", target_sums)
#     return df_final

def combine_results(baseline_df, your_df):
    """
    Combine baseline and your results into a single DataFrame
    """
    # Get all unique indices from both DataFrames, maintaining original order
    baseline_indices = baseline_df.index.tolist()
    your_indices = your_df.index.tolist()

    # Combine indices while preserving baseline order first
    all_indices = []
    for idx in baseline_indices:
        if idx not in all_indices:
            all_indices.append(idx)
    for idx in your_indices:
        if idx not in all_indices:
            all_indices.append(idx)

    # Get all unique columns - preserving the order correctly
    all_columns = []

    # First add all columns from baseline_df
    for col in baseline_df.columns:
        if col not in all_columns:
            all_columns.append(col)

    # Then add all columns from your_df
    for col in your_df.columns:
        if col not in all_columns:
            all_columns.append(col)

    # Create combined DataFrame with proper MultiIndex
    combined_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(all_indices),
                               columns=pd.MultiIndex.from_tuples(all_columns))

    # Fill in baseline data
    for idx in baseline_df.index:
        for col in baseline_df.columns:
            value = baseline_df.loc[idx, col]
            combined_df.loc[idx, col] = value if value != "-" and pd.notna(value) else "-"

    # Fill in your data
    for idx in your_df.index:
        for col in your_df.columns:
            value = your_df.loc[idx, col]
            combined_df.loc[idx, col] = value if value != "-" and pd.notna(value) else "-"

    # Fill any remaining NaN values with "-"
    combined_df = combined_df.fillna("-")

    return combined_df
def load_your_results(results_dir, method_name="InVirtuoGen", require_sim=True):
    """
    Load results from docking_..._thr*.csv.
    If require_sim is False, ignore sim and aggregate across all thr files into one column.
    Filtering:
      require_sim=True  -> sim > thr_val, QED > 0.6, SA < 4
      require_sim=False -> QED > 0.6, SA < 4
    """
    rx = re.compile(r"docking_(?P<target>[\w\-]+)_idx(?P<idx>\d+)_thr(?P<thr>[46])\.csv")
    files = glob.glob(os.path.join(results_dir, "**/docking_*_idx*_thr*.csv"), recursive=True)

    targets, idxs = set(), set()
    file_map = {}  # (target, idx) -> list of file paths
    for fn in files:
        m = rx.search(os.path.basename(fn))
        if not m:
            continue
        target, idx = m.group("target"), int(m.group("idx"))
        targets.add(target)
        idxs.add(idx)
        file_map.setdefault((target, idx), []).append(fn)

    targets = sorted(targets)
    idxs = sorted(idxs)

    # choose outward column labels
    if require_sim:
        thrs = ["4", "6"]
        col_labels = [f"δ = 0.{t}" for t in thrs]
    else:
        thrs = ["nosim"]
        col_labels = ["No sim"]

    # build container
    results = {t: {i: {thr: [] for thr in thrs} for i in idxs} for t in targets}

    # parse files
    for (target, idx), fns in file_map.items():
        if require_sim:
            # process per thr, keep sim constraint
            for fn in fns:
                m = rx.search(os.path.basename(fn))
                thr = m.group("thr")
                try:
                    df_temp = pd.read_csv(fn)
                    req = ['sim', 'qed', 'sa', 'seed', 'docking score']
                    if not all(c in df_temp.columns for c in req):
                        continue
                    thr_val = float(thr) / 10.0
                    df_filtered = df_temp[(df_temp['sim'] > thr_val) & (df_temp['qed'] > 0.6) & (df_temp['sa'] < 4)]
                    if df_filtered.empty:
                        continue
                    s = df_filtered.groupby("seed")["docking score"].max().tolist()
                    results[target][idx][thr].extend([-v for v in s])
                except Exception as e:
                    print(f"Error processing {fn}: {e}")
        else:
            # ignore sim, aggregate all thr files into 'nosim'
            bucket = "nosim"
            for fn in fns:
                try:
                    df_temp = pd.read_csv(fn)
                    req = ['qed', 'sa', 'seed', 'docking score']
                    if not all(c in df_temp.columns for c in req):
                        continue
                    df_filtered = df_temp[(df_temp['qed'] > 0.6) & (df_temp['sa'] < 4)]
                    if df_filtered.empty:
                        continue
                    s = df_filtered.groupby("seed")["docking score"].max().tolist()
                    results[target][idx][bucket].extend([-v for v in s])
                except Exception as e:
                    print(f"Error processing {fn}: {e}")

    # rows
    row_index = pd.MultiIndex.from_tuples([(t, i) for t in targets for i in idxs], names=['protein', 'idx'])
    # columns: keep a threshold level so downstream LaTeX code works unchanged
    columns = pd.MultiIndex.from_tuples([(lab, method_name) for lab in col_labels], names=['threshold', 'method'])
    df_final = pd.DataFrame(index=row_index, columns=columns)

    # fill
    for t in targets:
        for i in idxs:
            for thr, lab in zip(thrs, col_labels):
                scores = results[t][i][thr]
                if not scores:
                    val = "-"
                elif len(scores) == 1:
                    val = f"{scores[0]:.1f}"
                else:
                    m = np.mean(scores); s = np.std(scores, ddof=1)
                    val = f"{m:.1f}±{s:.1f}"
                df_final.loc[(t, i), (lab, method_name)] = val

    return df_final
def create_latex_table(
    df,
    caption=None,
    label="tab:docking_comparison",
    baseline_raw_df=None,
    seed_metrics=None,
    your_method=("InVirtuoGen", "InVirtuoGen"),
):
    if df is None:
        raise ValueError("create_latex_table: df is None")

    if caption is None:
        caption = (
           r"Docking scores (lower is better) averaged over 3 seeds. Bold indicates the best result per seed. Values in parentheses denote solutions that do not improve the docking score over the seed but still satisfy QED$>0.6$ and SA$<4$. For each seed, its docking score, the quantitative estimate of drug-likeness and synthetic accessibility is given."
        )

    # seed maps
    seed_score_map = {}
    if baseline_raw_df is not None:
        for tgt in baseline_raw_df["target"].unique():
            tdf = baseline_raw_df[baseline_raw_df["target"] == tgt].reset_index(drop=True)
            for idx in range(len(tdf)):
                seed_score_map[(tgt, idx)] = float(tdf.iloc[idx]["seed_score"])

    seed_qed_map, seed_sa_map = {}, {}
    if seed_metrics is not None:
        for k, (q, s) in seed_metrics.items():
            seed_qed_map[k] = float(q)
            seed_sa_map[k] = float(s)

    thresholds = df.columns.get_level_values(0).unique()
    methods_per_threshold = {thr: [m for t, m in df.columns if t == thr] for thr in thresholds}

    def parse_value(val):
        if val == "-" or pd.isna(val):
            return None, None
        s = str(val).strip()
        if "±" in s:
            a, b = s.split("±", 1)
            return float(a), float(b)
        return float(s), 0.0

    # precompute sums (exclusive and inclusive) and best-by-sum per threshold
    sums_excl, sums_incl = {}, {}
    best_per_threshold = {}
    for thr in thresholds:
        min_excl = (None, float("inf"))
        for method in methods_per_threshold[thr]:
            se = si = 0.0
            ce = ci = 0
            for idx in df.index:
                v = df.loc[idx, (thr, method)]
                mv, _ = parse_value(v)
                if mv is None:
                    continue
                ci += 1
                si += mv
                seed_score = seed_score_map.get(idx, None)
                if seed_score is None or mv <= seed_score:
                    ce += 1
                    se += mv
            sums_excl[(thr, method)] = se if ce > 0 else None
            sums_incl[(thr, method)] = si if ci > 0 else None

            if se is not None and se < min_excl[1]:
                min_excl = (method, se)
        best_per_threshold[thr] = min_excl[0]

    # open table
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begingroup",
        r"\setlength\tabcolsep{4pt}",
    ]

    # one left column, then centered X columns
    from collections import OrderedDict

    # method_col_count = sum(len(m) for m in methods_per_threshold.values())

    # col_spec = "@{}l "
    # for methods in methods_per_threshold.values():
    #     for m in methods:
    #         if m == "InVirtuoGen":
    #             col_spec += ">{\\columncolor{gray!20}}c "
    #         else:
    #             col_spec += "c "
    # col_spec += "@{}"
    method_col_count = sum(len(m) for m in methods_per_threshold.values())

    col_parts = ["@{}l"]
    for thr in thresholds:
        col_parts.append("|")  # vertical line before each threshold block
        for m in methods_per_threshold[thr]:
            if m == "InVirtuoGen":
                col_parts.append(">{\\columncolor{gray!20}}c")
            else:
                col_parts.append("c")
    col_parts.append("@{}")
    col_spec = " ".join(col_parts)

    latex_lines += [
        r"\begin{tabularx}{\linewidth}{" + col_spec + "}",
        r"\toprule"
    ]
    # header 1 (aligned): Protein with tiny note stacked; deltas centered
    header1 = [r"\shortstack{Protein \\ \tiny{(DS/QED/SA)}}"]
    for thr in thresholds:
        n_methods = len(methods_per_threshold[thr])
        thr_ltx = thr.replace("δ", r"$\delta$")
        header1.append(r"\multicolumn{" + str(n_methods) + "}{c}{" + thr_ltx + "}")
    latex_lines.append(" & ".join(header1) + r" \\")

    # cmidrules start at column 2
    cstart = 2
    parts = []
    for thr in thresholds:
        n = len(methods_per_threshold[thr])
        cend = cstart + n - 1
        parts.append(f"\\cmidrule(lr){{{cstart}-{cend}}}")
        cstart = cend + 1
    if parts:
        latex_lines.append(" ".join(parts))

    # header 2
    header2 = [""]
    for thr in thresholds:
        header2.extend(methods_per_threshold[thr])
    latex_lines.append(" & ".join(header2) + r" \\")
    latex_lines.append(r"\midrule")

    # body
    current_protein = None
    for (protein, idx), row in df.iterrows():
        seed_score = seed_score_map.get((protein, idx), None)
        seed_qed = seed_qed_map.get((protein, idx), None)
        seed_sa = seed_sa_map.get((protein, idx), None)

        # Find minimum value and its std dev per threshold among methods that beat the seed
        min_info_per_thr = {}  # {thr: (min_value, min_std, min_method)}
        for thr in thresholds:
            min_val = float("inf")
            min_std = 0.0
            min_method = None

            # First pass: find the minimum value among those that beat the seed
            for method in methods_per_threshold[thr]:
                v = row.get((thr, method), "-")

                mv, sv = parse_value(v)
                sv = 0 if method!="InVirtuoGen" else sv
                if mv is None:
                    continue
                if seed_score is not None and mv >= seed_score:
                    continue
                if mv < min_val:
                    min_val = mv
                    min_std = sv if sv is not None else 0.0
                    min_method = method

            if min_method is not None:
                min_info_per_thr[thr] = (min_val, min_std, min_method)

        # start protein block
        if protein != current_protein:
            current_protein = protein
            # protein_header = [protein] + [""] * method_col_count
            # latex_lines.append(" & ".join(protein_header) + r" \\")
            protein_header = [protein] + [""] * method_col_count
            latex_lines.append(" & ".join(protein_header) + r" \\")
            is_first_ds_row = True
        else:
            is_first_ds_row = False

        # DS/QED/SA left cell with in-row strut for separation (keeps vertical rule continuous)
        # if seed_score is not None and seed_qed is not None and seed_sa is not None:
        #     ds_text = f"{seed_score:.1f}/{seed_qed:.3f}/{seed_sa:.2f}"
        # elif seed_score is not None:
        #     ds_text = f"{seed_score:.1f}/-/-"
        # else:
        #     ds_text = "-"

        # strut = r"\rule{0pt}{1.0em} " if is_first_ds_row else ""
        # left = r"\tiny{" + strut + ds_text + "}"
        # row_cells = [left]
        # DS/QED/SA left cell with a leading strut on the first DS row of each protein
        if seed_score is not None and seed_qed is not None and seed_sa is not None:
            ds_text = f"{seed_score:.1f}/{seed_qed:.3f}/{seed_sa:.2f}"
        elif seed_score is not None:
            ds_text = f"{seed_score:.1f}/-/-"
        else:
            ds_text = "-"

        strut = r"\rule{0pt}{1.05em} " if is_first_ds_row else ""
        left = r"\tiny{" + strut + ds_text + "}"
        row_cells = [left]
        # DS/QED/SA string in the single left column for this index
        if seed_score is not None and seed_qed is not None and seed_sa is not None:
            left = r"\tiny{"+f"{seed_score:.1f}/{seed_qed:.3f}/{seed_sa:.2f}"+"}"
        elif seed_score is not None:
            left = r"\tiny{"+f"{seed_score:.1f}/-/-"+"}"
        else:
            left = "-"
        row_cells = [left]

        # method cells
        for thr in thresholds:
            min_info = min_info_per_thr.get(thr)

            for method in methods_per_threshold[thr]:
                v = row.get((thr, method), "-")
                if pd.isna(v) or v == "-":
                    cell = "-"
                else:
                    mv, sv = parse_value(v)
                    sv = None if method!="InVirtuoGen" else sv
                    beats_seed = (mv is not None and (seed_score is None or mv < seed_score))

                    # Determine if this value should be bold
                    should_bold = False
                    if beats_seed and min_info is not None:
                        min_val, min_std, _ = min_info
                        # Bold if within min ± min_std (only use the minimum's std dev)
                        if min_std > 0:
                            if mv <= min_val + min_std:
                                should_bold = True
                        else:
                            # No std dev for minimum, only bold the exact minimum
                            if mv == min_val:
                                should_bold = True

                    if method in your_method and mv is not None:
                        # format InVirtuo: math number, tiny std on same line
                        mean_str = r"\textbf{" + f"{mv:.1f}" + "}" if should_bold else f"{mv:.1f}"
                        cell = "$" + mean_str + "$"
                        if sv is not None:
                            cell += " " + "{\\tiny (" + "$\\pm " + f"{sv:.1f}" + "$" + ")}"
                    else:
                        # baselines: plain number; if a std is available, add tiny part
                        if mv is not None:
                            if sv is not None:
                                val = r"$\textbf{" + f"{mv:.1f}" + "}$" if should_bold else f"${mv:.1f}$"
                                cell = val + " " + "{\\tiny (" + "$\\pm " + f"{sv:.1f}" + "$" + ")}"
                            else:
                                cell = f"$\\textbf{{{mv:.1f}}}$" if should_bold else f"{mv:.1f}"

                    # parentheses if not an improvement
                    if mv is not None and seed_score is not None and mv > seed_score:
                        cell = f"({cell})"

                row_cells.append(cell)

        latex_lines.append(" & ".join(row_cells) + r" \\")

    # sum row: for all methods; InVirtuo also shows inclusive in parentheses
    latex_lines.append(r"\midrule")
    sum_row = ["\\multicolumn{1}{l}{Sum}"]
    for thr in thresholds:
        for method in methods_per_threshold[thr]:
            se = sums_excl[(thr, method)]
            si = sums_incl[(thr, method)]
            if se is None and si is None:
                cell = "-"
            elif method in your_method:

                cell = f"{se:.1f} ({si:.1f})" if se is not None and si is not None else "-"
            else:
                cell = f"{se:.1f}" if se is not None else "-"
            if best_per_threshold.get(thr) == method and se is not None:
                cell = f"$\\textbf{{{cell}}}$"
            sum_row.append(cell)
    latex_lines.append(" & ".join(sum_row) + r" \\")

    latex_lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\endgroup",
        r"\end{table}",
    ]
    return "\n".join(latex_lines)
# Main execution
if __name__ == "__main__":
    # Raw baseline data
    raw_baseline = """
Target | Seed score | GenMol | RetMol | Graph GA | GenMol | RetMol | Graph GA
parp1 | -7.3 | -10.6 | -9.0 | -8.3 | -10.4 | - | -8.6
 | -7.8 | -11.0 | -10.7 | -8.9 | -9.7 | - | -8.1
 | -8.2 | -11.3 | -10.9 | - | -9.2 | - | -
fa7 | -6.4 | -8.4 | -8.0 | -7.8 | -7.3 | -7.6 | -7.6
 | -6.7 | -8.4 | - | -8.2 | -7.6 | - | -7.6
 | -8.5 | - | - | - | - | - | -
5ht1b | -4.5 | -12.9 | -12.1 | -11.7 | -12.1 | - | -11.3
 | -7.6 | -12.3 | -9.0 | -12.1 | -12.0 | -10.0 | -12.0
 | -9.8 | -11.6 | - | - | -10.5 | - | -
braf | -9.3 | -10.8 | -11.6 | -9.8 | - | - | -
 | -9.4 | -10.8 | - | - | -9.7 | - | -
 | -9.8 | -10.6 | - | -11.6 | -10.5 | - | -10.4
jak2 | -7.7 | -10.2 | -8.2 | -8.7 | -9.3 | -8.1 | -
 | -8.0 | -10.0 | -9.0 | -9.2 | -9.4 | - | -9.2
 | -8.6 | -9.8 | - | - | - | - | -
"""

    # Parse baseline data
    baseline_raw_df = parse_baseline_table(raw_baseline)
    print("Parsed baseline data:")
    print(baseline_raw_df)
    print("\n" + "="*80 + "\n")

    # Convert to MultiIndex format
    baseline_df = baseline_df_to_multiindex(baseline_raw_df)
    print("Baseline data in MultiIndex format:")
    print(baseline_df)
    print("\n" + "="*80 + "\n")
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "nosim"], default="normal",
        help="normal: show δ=0.4 and δ=0.6 from the sim-constrained run only. nosim: single InVirtuoGen column without similarity constraint, no baselines.")
    parser.add_argument("--sim_results_dir", type=str, default=None,
        help="Directory of the run optimized with similarity constraint.")
    parser.add_argument("--no_sim_results_dir", type=str, default=None,
        help="Directory of the run optimized without similarity constraint.")
    parser.add_argument("--actives_csv", type=str,
        default="/home/bkaech/projects/public-release/in_virtuo_reinforce/docking/actives.csv")
    args = parser.parse_args()

    if args.mode == "normal":
        if not args.sim_results_dir:
            raise SystemExit("normal mode requires --sim_results_dir")
    else:  # nosim
        if not args.no_sim_results_dir:
            raise SystemExit("nosim mode requires --no_sim_results_dir")
    # args = parser.parse_args()
    # Load your results (update the path)
    # results_dir = args.results_dir
    # load sim-constrained results -> produces columns: ('δ = 0.4','InVirtuoGen'), ('δ = 0.6','InVirtuoGen')

    actives_csv = args.actives_csv
    baseline_df = baseline_df_to_multiindex(baseline_raw_df)

    if args.mode == "normal":
        # Only sim-constrained δ columns
        your_df_sim = load_your_results(args.sim_results_dir, method_name="InVirtuoGen", require_sim=True)

        # seeds for DS/QED/SA column
        seed_metrics = {}
        seed_metrics.update(extract_seed_metrics(args.sim_results_dir) or {})
        seed_metrics.update(load_seed_metrics_from_actives(actives_csv, baseline_raw_df) or {})

        combined_df = combine_results(baseline_df, your_df_sim)
        latex_table = create_latex_table(combined_df, baseline_raw_df=baseline_raw_df, seed_metrics=seed_metrics or None)
        print("LaTeX table:")
        print(latex_table)

    else:  # nosim
        # Single InVirtuoGen column, no baselines
        your_df_nosim = load_your_results(args.no_sim_results_dir, method_name="InVirtuoGen", require_sim=False)

        # DS/QED/SA source
        seed_metrics = load_seed_metrics_from_actives(actives_csv, baseline_raw_df) or {}

        latex_table = create_latex_table(your_df_nosim, baseline_raw_df=baseline_raw_df, seed_metrics=seed_metrics or None)
        print("LaTeX table (No sim only):")
        print(latex_table)