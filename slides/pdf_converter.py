import subprocess
import tempfile
import os


import subprocess
import tempfile
import os


def convert_pdf_to_markdown(pdf_path: str) -> str:
    """
    使用 OLMoCR pipeline 將單一 PDF 轉換成 Markdown，並回傳文字。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # 加上 --cpu
            result = subprocess.run(
                [
                    "python",
                    "-m", "olmocr.pipeline",
                    tmpdir,
                    "--markdown",
                    "--pdfs", pdf_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            # 找到輸出的 markdown
            md_files = [f for f in os.listdir(tmpdir) if f.endswith(".md")]
            if not md_files:
                return f"❌ 沒有產生 Markdown 輸出\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

            md_path = os.path.join(tmpdir, md_files[0])
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read()

        except subprocess.CalledProcessError as e:
            return f"❌ OLMoCR 轉換失敗：\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
