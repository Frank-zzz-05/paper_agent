"""生成一个合法的样例 PDF（tests/data/sample.pdf），用于离线加载测试。

内容为一段虚构的英文论文文本，纯 ASCII（Helvetica 字体），不依赖任何第三方库。
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "sample.pdf"

PARAGRAPHS = [
    "Attention Is All You Need: A Novel Transformer Architecture for Sequence Modeling",
    "Abstract: We propose the Transformer, a novel architecture based solely on attention "
    "mechanisms, dispensing with recurrence and convolutions entirely. Unlike recurrent models "
    "that process tokens sequentially, the Transformer processes all positions in parallel and "
    "computes relationships between any two positions with a constant number of operations.",
    "Introduction: Sequence transduction models have traditionally relied on recurrent or "
    "convolutional networks. However, their sequential nature inhibits parallelization and makes "
    "long-range dependencies difficult to capture. The core innovation of our work is replacing "
    "the recurrence entirely with multi-head self-attention.",
    "Method: The Transformer uses an encoder-decoder structure. Each layer applies multi-head "
    "self-attention followed by position-wise feed-forward networks, with residual connections "
    "and layer normalization. We introduce sinusoidal positional encodings to inject order "
    "information. Scaled dot-product attention is computed as softmax(QK^T/sqrt(dk))V.",
    "Experiments: We train on the WMT 2014 English-to-German and English-to-French translation "
    "tasks. Our model achieves a BLEU score of 28.4 on En-De, surpassing the previous best "
    "results while requiring significantly less training time than recurrent baselines. "
    "We also evaluate on English constituency parsing with good results.",
    "Conclusion: The Transformer is the first transduction model relying entirely on "
    "self-attention, and it can be trained significantly faster than architectures based on "
    "recurrent or convolutional layers. We hope this work encourages more attention-based "
    "models across the field.",
]


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf() -> bytes:
    lines = []
    lines.append("BT")
    lines.append("/F1 11 Tf")
    lines.append("54 738 Td")
    lines.append("17 TL")
    for para in PARAGRAPHS:
        # 简单折行：每行约 92 字符
        for i in range(0, len(para), 92):
            lines.append(f"({_escape(para[i : i + 92])}) Tj")
            lines.append("T*")
        lines.append("T*")
    lines.append("ET")
    stream_content = "\n".join(lines).encode("ascii")
    stream_len = len(stream_content)

    def obj(num: int, body: str) -> str:
        return f"{num} 0 obj\n{body}\nendobj\n"

    body1 = "<< /Type /Catalog /Pages 2 0 R >>"
    body2 = "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    body3 = (
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    body4 = f"<< /Length {stream_len} >>\nstream\n{stream_content.decode('ascii')}\nendstream"
    body5 = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    parts = [header]
    offsets = []
    running = len(header)
    for body in (body1, body2, body3, body4, body5):
        offsets.append(running)
        obj_bytes = obj(len(offsets), body).encode("ascii")
        parts.append(obj_bytes)
        running += len(obj_bytes)

    xref_pos = running
    xref = f"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    parts.append(xref.encode("ascii"))
    parts.append(trailer.encode("ascii"))
    return b"".join(parts)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(build_pdf())
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
