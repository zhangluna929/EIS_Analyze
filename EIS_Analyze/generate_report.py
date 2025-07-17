import pandas as pd
from pathlib import Path
import datetime
import base64

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<title>EIS Batch Analysis Report</title>
<style>
 body {font-family: Arial, sans-serif; margin: 20px;}
 h1 {text-align: center;}
 table {border-collapse: collapse; width: 100%; margin-bottom: 40px;}
 th, td {border: 1px solid #aaa; padding: 6px; text-align: center; font-size: 12px;}
 th {background-color: #f0f0f0;}
 img {max-width: 600px; margin: 10px auto; display: block; border:1px solid #ccc;}
</style>
</head>
<body>
<h1>EIS Batch Analysis Report</h1>
<p>Generated: {generated_time}</p>
{table_html}
{images_html}
</body>
</html>"""


def img_to_base64(img_path: Path):
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode('utf-8')
    suffix = img_path.suffix.lstrip('.')
    return f"data:image/{suffix};base64,{b64}"


def generate_report(csv_path: Path = Path('batch_fit_results.csv'), output_html: Path = Path('analysis_report.html')):
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run batch analysis first.")
        return

    df = pd.read_csv(csv_path)

    # build HTML table
    table_html = df.to_html(index=False, float_format=lambda x: f"{x:.4e}" if isinstance(x, float) else x)

    # build images section
    images_parts = []
    for fname in df['filename']:
        img_path = Path(fname).stem + '_analysis.png'
        img_data = img_to_base64(Path(img_path))
        if img_data:
            images_parts.append(f"<h3>{fname}</h3><img src='{img_data}' alt='{fname} analysis'>")
    images_html = "\n".join(images_parts)

    html_content = HTML_TEMPLATE.format(generated_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                                        table_html=table_html,
                                        images_html=images_html)

    output_html.write_text(html_content, encoding='utf-8')
    print(f"Report generated: {output_html}")


if __name__ == '__main__':
    generate_report() 