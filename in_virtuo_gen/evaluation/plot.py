import pandas as pd
import matplotlib.pyplot as plt
import re
import sys


def extract_value_and_std(cell):
    """Extract both mean value and standard deviation from a cell containing '±' notation."""
    if isinstance(cell, str) and '±' in cell:
        parts = cell.split('±')
        return float(parts[0].strip()), float(parts[1].strip())
    return float(cell) if isinstance(cell, (int, float)) else cell, 0.0


def read_markdown_table(file_path):
    """Read and parse a markdown file containing a table."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            markdown_text = file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    lines = markdown_text.split('\n')
    data = []
    headers = None
    current_main_method = None

    for line in lines:
        if '|' in line:
            if '-|-' in line:
                continue

            cells = [cell.strip() for cell in line.split('|')[1:-1]]

            if not cells or not any(cells):
                continue

            if 'Method' in cells[0]:
                headers = cells
                continue

            if cells[0]:
                if 'GenMol' in cells[0] or 'FragMol' in cells[0] or 'SAFE-GPT' in cells[0]:
                    current_main_method = cells[0].split('(')[0].strip()
                    if len(cells) > 1 and any(cells[1:]):
                        processed_cells = [cells[0]]
                        values_and_stds = [extract_value_and_std(cell) for cell in cells[1:]]
                        values = [v[0] for v in values_and_stds]
                        stds = [v[1] for v in values_and_stds]
                        processed_cells.extend(values)
                        processed_cells.extend(stds)
                        data.append(processed_cells)
                else:
                    method_name = f"{current_main_method} {cells[0]}"
                    processed_cells = [method_name]
                    values_and_stds = [extract_value_and_std(cell) for cell in cells[1:]]
                    values = [v[0] for v in values_and_stds]
                    stds = [v[1] for v in values_and_stds]
                    processed_cells.extend(values)
                    processed_cells.extend(stds)
                    data.append(processed_cells)
            else:
                if current_main_method and len(cells) > 1:
                    method_name = f"{current_main_method} {cells[1]}"
                    processed_cells = [method_name]
                    values_and_stds = [extract_value_and_std(cell) for cell in cells[1:]]
                    values = [v[0] for v in values_and_stds]
                    stds = [v[1] for v in values_and_stds]
                    processed_cells.extend(values)
                    processed_cells.extend(stds)
                    data.append(processed_cells)

    # Create extended headers for standard deviations
    value_headers = headers[1:]
    std_headers = [f"{h} std" for h in value_headers]
    extended_headers = [headers[0]] + value_headers + std_headers

    # Create DataFrame
    df = pd.DataFrame(data, columns=extended_headers)
    return df


def get_method_data(df, method_name):
    """Extract data for a specific method."""
    if method_name == 'SAFE-GPT':
        return df[df['Method'] == 'SAFE-GPT']
    else:
        return df[df['Method'].str.startswith(method_name)]


def create_plots(df):
    """Create three comparison plots in a single row."""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # Create figure with more space at the top
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Molecular Generation Performance Metrics', fontsize=16, y=0.98)

    # Create a grid of subplots in a single row
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    axes = []
    for i in range(3):
        axes.append(fig.add_subplot(gs[0, i]))

    colors = {'SAFE-GPT': 'red', 'GenMol': 'blue', 'FragMol': 'green'}
    markers = {'SAFE-GPT': 'o', 'GenMol': 's', 'FragMol': '^'}

    safe_gpt_data = get_method_data(df, 'SAFE-GPT')
    genmol_data = get_method_data(df, 'GenMol')
    fragmol_data = get_method_data(df, 'FragMol')

    print("\nNumber of rows for each method:")
    print(f"SAFE-GPT: {len(safe_gpt_data)}")
    print(f"GenMol: {len(genmol_data)}")
    print(f"FragMol: {len(fragmol_data)}")

    # Plot 1: Diversity vs Quality
    ax = axes[0]
    # Plot main data point
    ax.scatter(safe_gpt_data['Diversity'], safe_gpt_data['Quality (%)'],
               color=colors['SAFE-GPT'], label='SAFE-GPT', marker='o', s=100)
    # Plot error bars with lighter color and dashed lines
    ax.errorbar(safe_gpt_data['Diversity'], safe_gpt_data['Quality (%)'],
                xerr=safe_gpt_data['Diversity std'], yerr=safe_gpt_data['Quality (%) std'],
                color=colors['SAFE-GPT'], alpha=0.2, ls=(0, (5, 5)), capsize=3, markersize=0)

    if not genmol_data.empty:
        # Plot main line and points
        ax.plot(genmol_data['Diversity'], genmol_data['Quality (%)'],
                color=colors['GenMol'], label='GenMol', marker='s', markersize=8, linewidth=3)
        # Plot error bars with lighter color and dashed lines
        ax.errorbar(genmol_data['Diversity'], genmol_data['Quality (%)'],
                    xerr=genmol_data['Diversity std'], yerr=genmol_data['Quality (%) std'],
                    color=colors['GenMol'], alpha=0.2, ls='--', capsize=3, markersize=0)
    if not fragmol_data.empty:
        # Plot main line and points
        ax.plot(fragmol_data['Diversity'], fragmol_data['Quality (%)'],
                color=colors['FragMol'], label='FragMol-80M', marker='^', markersize=8, linewidth=2)
        # Plot error bars with lighter color and dashed lines
        ax.errorbar(fragmol_data['Diversity'], fragmol_data['Quality (%)'],
                    xerr=fragmol_data['Diversity std'], yerr=fragmol_data['Quality (%) std'],
                    color=colors['FragMol'], alpha=0.2, ls='--', capsize=3, markersize=0)

    ax.set_xlabel('Diversity')
    ax.set_ylabel('Quality (%)')
    ax.set_title('Diversity vs Quality')
    ax.legend()

    # Plot 2: Validity vs Diversity
    ax = axes[1]
    # Plot main data point
    ax.scatter(safe_gpt_data['Diversity'], safe_gpt_data['Validity (%)'],
               color=colors['SAFE-GPT'], label='SAFE-GPT', marker='o', s=100)
    # Plot error bars with lighter color and dashed lines
    ax.errorbar(safe_gpt_data['Diversity'], safe_gpt_data['Validity (%)'],
                xerr=safe_gpt_data['Diversity std'], yerr=safe_gpt_data['Validity (%) std'],
                color=colors['SAFE-GPT'], alpha=0.2, ls='--', capsize=3, markersize=0)

    if not genmol_data.empty:
        # Plot main line and points
        ax.plot(genmol_data['Diversity'], genmol_data['Validity (%)'],
                color=colors['GenMol'], label='GenMol', marker='s', markersize=8)
        # Plot error bars with lighter color and dashed lines
        ax.errorbar(genmol_data['Diversity'], genmol_data['Validity (%)'],
                    xerr=genmol_data['Diversity std'], yerr=genmol_data['Validity (%) std'],
                    color=colors['GenMol'], alpha=0.2, ls='--', capsize=3, markersize=0)
    if not fragmol_data.empty:
        # Plot main line and points
        ax.plot(fragmol_data['Diversity'], fragmol_data['Validity (%)'],
                color=colors['FragMol'], label='FragMol-80M', marker='^', markersize=8)
        # Plot error bars with lighter color and dashed lines
        ax.errorbar(fragmol_data['Diversity'], fragmol_data['Validity (%)'],
                    xerr=fragmol_data['Diversity std'], yerr=fragmol_data['Validity (%) std'],
                    color=colors['FragMol'], alpha=0.2, ls='--', capsize=3, markersize=0)

    ax.set_xlabel('Diversity')
    ax.set_ylabel('Validity (%)')
    ax.set_title('Validity vs Diversity')
    ax.legend()

    # Plot 3: Diversity vs Uniqueness
    ax = axes[2]
    # Plot main data point
    ax.scatter(safe_gpt_data['Diversity'], safe_gpt_data['Uniqueness (%)'],
               color=colors['SAFE-GPT'], label='SAFE-GPT', marker='o', s=100)
    # Plot error bars with lighter color and dashed lines
    ax.errorbar(safe_gpt_data['Diversity'], safe_gpt_data['Uniqueness (%)'],
                xerr=safe_gpt_data['Diversity std'], yerr=safe_gpt_data['Uniqueness (%) std'],
                color=colors['SAFE-GPT'], alpha=0.2, ls='--', capsize=3, markersize=0)

    if not genmol_data.empty:
        # Plot main line and points
        ax.plot(genmol_data['Diversity'], genmol_data['Uniqueness (%)'],
                color=colors['GenMol'], label='GenMol', marker='s', markersize=8)
        # Plot error bars with lighter color and dashed lines
        ax.errorbar(genmol_data['Diversity'], genmol_data['Uniqueness (%)'],
                    xerr=genmol_data['Diversity std'], yerr=genmol_data['Uniqueness (%) std'],
                    color=colors['GenMol'], alpha=0.2, ls='--', capsize=3, markersize=0)
    if not fragmol_data.empty:
        # Plot main line and points
        ax.plot(fragmol_data['Diversity'], fragmol_data['Uniqueness (%)'],
                color=colors['FragMol'], label='FragMol-80M', marker='^', markersize=8)
        # Plot error bars with lighter color and dashed lines
        ax.errorbar(fragmol_data['Diversity'], fragmol_data['Uniqueness (%)'],
                    xerr=fragmol_data['Diversity std'], yerr=fragmol_data['Uniqueness (%) std'],
                    color=colors['FragMol'], alpha=0.2, ls='--', capsize=3, markersize=0)

    ax.set_xlabel('Diversity')
    ax.set_ylabel('Uniqueness (%)')
    ax.set_title('Diversity vs Uniqueness')
    ax.legend()

    plt.show()


def main():
    """Main function to run the script."""
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_markdown_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = read_markdown_table(file_path)
    print("\nDataFrame head:")
    print(df.head())
    print("\nDataFrame shape:", df.shape)
    create_plots(df)


if __name__ == "__main__":
    main()