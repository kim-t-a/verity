"""Builders for synthetic DOCX, XLSX and PDF documents.

The readers are tested against two kinds of input: the real sample corpus
(which proves the readers work on documents a real producer wrote) and the
documents built here (which prove they behave correctly on shapes the sample
corpus does not happen to contain -- shared strings, wide tables, nested
tables, page trees, an empty workbook).

Everything is generated in memory or into a temporary directory. Nothing here
writes to ``mc3-corpus``, which is read-only reference data.
"""

from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------


def zip_bytes(parts: dict[str, str | bytes]) -> bytes:
    """Pack named parts into a ZIP, the way an OOXML document is packed."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in parts.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def column_letters(index: int) -> str:
    """Turn a zero-based column index into spreadsheet letters (0 -> A, 26 -> AA)."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


# --------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------

_WORD_NAMESPACE = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
)


def docx_paragraph(text: str) -> str:
    return f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>"


def docx_table(rows: list[list[str]]) -> str:
    body = ""
    for row in rows:
        cells = "".join(
            f"<w:tc>{docx_paragraph(cell)}</w:tc>" for cell in row
        )
        body += f"<w:tr>{cells}</w:tr>"
    return f"<w:tbl>{body}</w:tbl>"


def docx_bytes(body_xml: str, *, doctype: str = "") -> bytes:
    """Build a DOCX containing exactly the given body markup.

    ``doctype`` exists so a test can inject a document type definition and
    prove the reader refuses it rather than expanding it.
    """
    document = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        f"{doctype}"
        f"<w:document {_WORD_NAMESPACE}><w:body>{body_xml}</w:body></w:document>"
    )
    # A styles part is included deliberately: it is large, it is not content,
    # and a reader that walked the archive instead of naming its part would
    # pull it in.
    styles = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        f"<w:styles {_WORD_NAMESPACE}><w:style w:type='paragraph' w:styleId='Heading1'>"
        "<w:name w:val='heading 1'/></w:style></w:styles>"
    )
    return zip_bytes({"word/document.xml": document, "word/styles.xml": styles})


# --------------------------------------------------------------------------
# XLSX
# --------------------------------------------------------------------------

_SPREADSHEET_NAMESPACE = (
    'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
)
_RELATIONSHIP_NAMESPACE = (
    'xmlns="http://schemas.openxmlformats.org/package/2006/relationships"'
)
_OFFICE_RELATIONSHIP_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)


def _shared_string_pool(sheets: list[tuple[str, list[list[object]]]]) -> list[str]:
    pool: list[str] = []
    for _, rows in sheets:
        for row in rows:
            for cell in row:
                if isinstance(cell, str) and cell not in pool:
                    pool.append(cell)
    return pool


def _cell_markup(reference: str, value: object, shared: list[str] | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{reference}"><v>{value}</v></c>'
    text = str(value)
    if shared is None:
        return f'<c r="{reference}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'
    return f'<c r="{reference}" t="s"><v>{shared.index(text)}</v></c>'


def _worksheet_xml(rows: list[list[object]], shared: list[str] | None) -> str:
    body = ""
    for row_index, row in enumerate(rows, start=1):
        cells = "".join(
            _cell_markup(f"{column_letters(column_index)}{row_index}", value, shared)
            for column_index, value in enumerate(row)
        )
        body += f'<row r="{row_index}">{cells}</row>'
    return f"<worksheet {_SPREADSHEET_NAMESPACE}><sheetData>{body}</sheetData></worksheet>"


def xlsx_bytes(
    sheets: list[tuple[str, list[list[object]]]],
    *,
    use_shared_strings: bool = False,
    include_relationships: bool = True,
) -> bytes:
    """Build an XLSX from ``[(sheet name, rows), ...]``.

    ``use_shared_strings`` selects between the two ways a workbook can store
    text, because a reader that handles only one of them silently returns an
    empty column for every workbook that uses the other.
    """
    shared = _shared_string_pool(sheets) if use_shared_strings else None

    parts: dict[str, str | bytes] = {}
    sheet_entries = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index + 1}" r:id="rId{index + 1}"/>'
        for index, (name, _) in enumerate(sheets)
    )
    parts["xl/workbook.xml"] = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        f'<workbook {_SPREADSHEET_NAMESPACE} '
        f'xmlns:r="{_OFFICE_RELATIONSHIP_NAMESPACE}">'
        f"<sheets>{sheet_entries}</sheets></workbook>"
    )

    if include_relationships:
        relationships = "".join(
            f'<Relationship Id="rId{index + 1}" '
            f'Type="{_OFFICE_RELATIONSHIP_NAMESPACE}/worksheet" '
            f'Target="worksheets/sheet{index + 1}.xml"/>'
            for index in range(len(sheets))
        )
        parts["xl/_rels/workbook.xml.rels"] = (
            "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
            f"<Relationships {_RELATIONSHIP_NAMESPACE}>{relationships}</Relationships>"
        )

    for index, (_, rows) in enumerate(sheets, start=1):
        parts[f"xl/worksheets/sheet{index}.xml"] = _worksheet_xml(rows, shared)

    if shared is not None:
        items = "".join(f"<si><t>{escape(text)}</t></si>" for text in shared)
        parts["xl/sharedStrings.xml"] = (
            "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
            f'<sst {_SPREADSHEET_NAMESPACE} count="{len(shared)}" '
            f'uniqueCount="{len(shared)}">{items}</sst>'
        )

    return zip_bytes(parts)


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


def pdf_bytes(
    *,
    text: str = "Synthetic document",
    encrypted: bool = False,
    compressed: bool = False,
) -> bytes:
    """Build a small single-page PDF.

    Written by hand rather than with a library so the test controls the exact
    structure: whether the trailer declares ``/Encrypt``, whether the content
    stream is Flate-compressed, and whether a page tree is present at all.
    """
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    if compressed:
        import zlib

        stream_body = zlib.compress(content)
        stream_header = b"<< /Length %d /Filter /FlateDecode >>" % len(stream_body)
    else:
        stream_body = content
        stream_header = b"<< /Length %d >>" % len(stream_body)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        stream_header + b"\nstream\n" + stream_body + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    for number, body in enumerate(objects, start=1):
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    trailer = b"<< /Size 6 /Root 1 0 R"
    if encrypted:
        trailer += b"/Encrypt << /Filter /Standard /R 6 /V 5 /Length 256 >>"
    trailer += b" >>"
    out += b"trailer\n" + trailer + b"\n%%EOF\n"
    return bytes(out)
