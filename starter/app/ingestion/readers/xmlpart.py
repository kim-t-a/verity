"""Loading the XML parts inside an OOXML document, safely.

DOCX and XLSX are ZIP archives full of XML. Parsing that XML with the standard
library is fine, but ``xml.etree`` will happily expand a document-type
definition, and a small file can define entities that expand exponentially --
the "billion laughs" shape -- turning a few hundred bytes into gigabytes of
memory. Corpus files are untrusted input, so this is a real hazard rather than a
theoretical one.

The guard is to reject any part containing a DTD. OOXML produced by Word, Excel
or any mainstream library never has one, so refusing a document that does costs
us nothing legitimate while removing the attack entirely. Rejecting is also
preferable to silently skipping: the caller turns this into a recorded failure,
so a rejected part is visible rather than absent.
"""

from __future__ import annotations

import zipfile
from xml.etree import ElementTree

from config import MAX_ZIP_MEMBER_BYTES


class XmlPartError(Exception):
    """A ZIP member could not be read as XML."""


def _document_type_is_present(raw: bytes) -> bool:
    """Detect a DTD or entity declaration in the first part of the bytes.

    Checked on the raw bytes before parsing, because by the time a parser has
    seen an entity declaration the expansion has already begun. The search is
    limited to the head of the document: a DTD must appear before the root
    element, so anything later is just text that happens to contain the word.
    """
    head = raw[:4096].upper()
    return b"<!DOCTYPE" in head or b"<!ENTITY" in head


def read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    """Read one member of a ZIP, refusing anything implausibly large.

    The declared uncompressed size is checked before decompressing, because a
    ZIP bomb's whole trick is a tiny compressed size paired with an enormous
    uncompressed one -- a check made after decompression would already have
    lost.
    """
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise XmlPartError(f"{name} is missing from the archive") from exc

    if info.file_size > MAX_ZIP_MEMBER_BYTES:
        raise XmlPartError(
            f"{name} declares {info.file_size} bytes, above the {MAX_ZIP_MEMBER_BYTES} cap"
        )

    with archive.open(info) as handle:
        # Read one byte past the cap: if that succeeds the declared size lied.
        raw = handle.read(MAX_ZIP_MEMBER_BYTES + 1)
    if len(raw) > MAX_ZIP_MEMBER_BYTES:
        raise XmlPartError(f"{name} exceeded the {MAX_ZIP_MEMBER_BYTES} byte cap while reading")
    return raw


def parse_member(archive: zipfile.ZipFile, name: str) -> ElementTree.Element:
    """Read and parse one XML member of a ZIP archive."""
    raw = read_member(archive, name)
    if _document_type_is_present(raw):
        raise XmlPartError(f"{name} contains a document type definition, which is refused")
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise XmlPartError(f"{name} is not well-formed XML: {exc}") from exc


def open_archive(path) -> zipfile.ZipFile:
    """Open a DOCX or XLSX as a ZIP, reporting a corrupt container clearly."""
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise XmlPartError(f"not a readable ZIP container: {exc}") from exc
    except OSError as exc:
        raise XmlPartError(f"could not open the container: {exc}") from exc


def local_name(tag: str) -> str:
    """The element name without its ``{namespace}`` prefix.

    OOXML documents in the wild use the transitional and strict namespace
    variants interchangeably, and a producer is free to pick its own prefix.
    Comparing local names sidesteps all of that: the prefix is a spelling
    detail, whereas ``t`` or ``tbl`` is the actual element being looked for.
    """
    return tag.rpartition("}")[2]
