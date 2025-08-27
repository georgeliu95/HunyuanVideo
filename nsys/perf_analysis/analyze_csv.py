import argparse
import pandas as pd
import os


KEEP_SRC_COLUMNS = ["kernel_id", "Name", "Start Timestamp", "Context", "Duration Value"]
ALL_KERNEL_TYPES = ("fMHA", "GEMM", "GEMV", "NCCL", "Norm", "Reformat", "Reduce", "Copy", "Elementwise", "Others")


def determine_kernel_type(row):
    """
    Determine kernel type based on row data
    Return value must be one of ALL_KERNEL_TYPES
    """
    # Assume CSV has a column containing kernel name information (could be 'name', 'kernel_name', 'op_name', etc.)
    # You can adjust based on actual CSV column names
    
    # Try to get kernel name from possible columns
    kernel_name = row["Name"]
    
    # Ignore capital case when matching
    kernel_name_lower = kernel_name.lower()
    if any(keyword in kernel_name_lower for keyword in ['fmha', 'flash', 'attention']):
        return "fMHA"
    elif any(keyword in kernel_name_lower for keyword in ['gemm', 'nvjet']):
        return "GEMM"
    elif any(keyword in kernel_name_lower for keyword in ['gemv']):
        return "GEMV"
    elif any(keyword in kernel_name_lower for keyword in ['nccl']):
        return "NCCL"
    elif any(keyword in kernel_name_lower for keyword in ['layernorm', 'layer_norm', 'norm']):
        return "Norm"
    elif any(keyword in kernel_name_lower for keyword in ['reformat', 'reshape', 'transpose', 'permute', 'nchw', 'nhwc']):
        return "Reformat"
    elif any(keyword in kernel_name_lower for keyword in ['reduce']):
        return "Reduce"
    elif any(keyword in kernel_name_lower for keyword in ['copy']):
        return "Copy"
    elif any(keyword in kernel_name_lower for keyword in ['elementwise']):
        return "Elementwise"
    else:
        return "Others"


# Create a function to output to both console and file
def dual_print(text, file_handle):
    print(text)
    file_handle.write(text + "\n")

# Function to convert DataFrame to markdown table
def df_to_markdown(df):
    # Create header
    headers = ["Type"] + df.columns.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|" + "|".join([" --- " for _ in headers]) + "|"
    
    # Create rows
    rows = []
    for idx, row in df.iterrows():
        row_values = []
        for col_name, val in zip(df.columns, row.values):
            if col_name == 'Count':
                # Format Count as integer
                row_values.append(str(int(val)))
            elif col_name == 'Percentage':
                # Format Percentage with % symbol
                row_values.append(f"{val}%")
            else:
                # Other columns as is
                row_values.append(str(val))
        rows.append("| " + " | ".join([idx] + row_values) + " |")
    
    return "\n".join([header_line, separator_line] + rows)


def process_csv(csv_file):
    df = pd.read_csv(csv_file)
    df = df[KEEP_SRC_COLUMNS]
    # 将 "Duration Value" 列重命名为 "Duration"
    df = df.rename(columns={"Duration Value": "Duration"})
    # Add Type column
    df['Type'] = df.apply(determine_kernel_type, axis=1)

    return df


def analyze_single_csv(df, file_handle, filename):
    """
    分析单个CSV文件并输出结果到文件句柄
    """
    # Check if Duration column exists
    if 'Duration' not in df.columns:
        dual_print(f"**Error:** 'Duration' column not found in {filename}.", file_handle)
        dual_print("Available columns: " + str(list(df.columns)), file_handle)
        return
    
    # Convert Duration to numeric, handling any non-numeric values
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    # Remove rows with NaN duration values
    df_clean = df.dropna(subset=['Duration'])
    
    if df_clean.empty:
        dual_print(f"**Error:** No valid duration data found in {filename}.", file_handle)
        return
    
    # Group by Type and calculate statistics
    stats_by_type = df_clean.groupby('Type')['Duration'].agg([
        ('Count', 'count'),
        ('Sum', 'sum'),
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Std', 'std'),
        ('25%', lambda x: x.quantile(0.25)),
        ('75%', lambda x: x.quantile(0.75))
    ]).round(3)
    
    # Calculate percentage of total duration for each type
    total_duration = df_clean['Duration'].sum()
    stats_by_type['Percentage'] = (stats_by_type['Sum'] / total_duration * 100).round(2)
    
    # Sort by sum in descending order
    stats_by_type = stats_by_type.sort_values('Sum', ascending=False)
    
    dual_print("### Detailed Duration Statistics by Kernel Type", file_handle)
    dual_print("", file_handle)
    
    # Convert to markdown table format
    markdown_table = df_to_markdown(stats_by_type)
    dual_print(markdown_table, file_handle)
    dual_print("", file_handle)
    
    dual_print("### Summary", file_handle)
    dual_print("", file_handle)
    
    # Top 3 types by total duration
    dual_print("#### Top 3 Kernel Types by Total Duration", file_handle)
    dual_print("", file_handle)
    top_3 = stats_by_type.head(3)
    for idx, (kernel_type, row) in enumerate(top_3.iterrows(), 1):
        dual_print(f"{idx}. **{kernel_type}**: {row['Sum']:.3f} ({row['Percentage']:.2f}% of total)", file_handle)
    dual_print("", file_handle)
    
    # Overall statistics
    dual_print("#### Overall Statistics", file_handle)
    dual_print("", file_handle)
    dual_print(f"- **Total Duration**: {total_duration:.3f}", file_handle)
    dual_print(f"- **Total Kernel Calls**: {len(df_clean)}", file_handle)
    dual_print(f"- **Average Duration per Call**: {df_clean['Duration'].mean():.3f}", file_handle)
    dual_print(f"- **Median Duration per Call**: {df_clean['Duration'].median():.3f}", file_handle)
    dual_print("", file_handle)
    
    # Performance insights
    dual_print("#### Performance Insights", file_handle)
    dual_print("", file_handle)
    max_type = stats_by_type.index[0]  # Type with highest total duration
    max_avg_type = stats_by_type.sort_values('Mean', ascending=False).index[0]  # Type with highest avg duration
    most_frequent_type = stats_by_type.sort_values('Count', ascending=False).index[0]  # Most frequent type
    
    dual_print(f"- **Most time-consuming type (total)**: {max_type} ({stats_by_type.loc[max_type, 'Percentage']:.2f}%)", file_handle)
    dual_print(f"- **Highest average duration**: {max_avg_type} ({stats_by_type.loc[max_avg_type, 'Mean']:.3f})", file_handle)
    dual_print(f"- **Most frequent kernel type**: {most_frequent_type} ({stats_by_type.loc[most_frequent_type, 'Count']} calls)", file_handle)


def main(args):
    if not os.path.exists(args.csv):
        print(f"Error: CSV file {args.csv} does not exist")
        return
    if os.path.isdir(args.csv):
        # 处理目录中的多个CSV文件
        csv_files = [f for f in os.listdir(args.csv) if f.endswith(".csv")]
        if not csv_files:
            print(f"Error: No CSV files found in directory {args.csv}")
            return
        
        # 确保输出目录存在，不存在则创建
        input_filename = os.path.splitext(os.path.basename(csv_files[0]))[0].partition(".s40d20.")[0].partition(".d20.")[0]
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = f"{args.output_dir}/{input_filename}.combined_analysis.md"
        
        # Create output file and output to both console and file
        with open(output_filename, 'w', encoding='utf-8') as f:
            dual_print("# Combined Duration Statistics by Kernel Type", f)
            dual_print("", f)
            
            # 处理每个CSV文件
            for csv_file in csv_files:
                csv_path = os.path.join(args.csv, csv_file)
                input_filename = os.path.splitext(csv_file)[0]
                
                dual_print(f"## Analysis for {input_filename}", f)
                dual_print("", f)
                
                df = process_csv(csv_path)
                analyze_single_csv(df, f, input_filename)
                dual_print("", f)
                dual_print("---", f)
                dual_print("", f)
    
    else:
        df = process_csv(args.csv)
        # Generate output filename based on input CSV filename
        input_filename = os.path.splitext(os.path.basename(args.csv))[0]
        # 确保输出目录存在，不存在则创建
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = f"{args.output_dir}/{input_filename}.md"

        # Create output file and output to both console and file
        with open(output_filename, 'w', encoding='utf-8') as f:
            dual_print("# Duration Statistics by Kernel Type", f)
            dual_print("", f)
            
            analyze_single_csv(df, f, input_filename)
    
    print(f"\nAnalysis complete! Results also saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", type=str, help="The CSV file to analyze")
    parser.add_argument("--output-dir", type=str, default="./results", help="The directory to save the plots")
    args = parser.parse_args()
    main(args)