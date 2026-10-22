import type { Citation } from "@/types";

/**
 * Answer text is model output, so it may or may not contain `[n]` markers.
 * A marker is turned into a link to evidence only when `n` matches a citation
 * the backend actually returned. Anything else stays as literal text: a
 * pointer to evidence that does not exist would be an invented citation.
 */
export type InlineSegment =
  | { kind: "text"; text: string }
  | { kind: "citation"; numeral: number };

const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/** Live answers carry `ordinal`; history citations do not. */
export function hasNumeralMapping(citations: Citation[]): boolean {
  return citations.some((citation) => typeof citation.ordinal === "number");
}

export function numeralOf(citation: Citation): number | null {
  return typeof citation.ordinal === "number" ? citation.ordinal : null;
}

export function citationAt(
  citations: Citation[],
  numeral: number
): Citation | undefined {
  return citations.find((citation) => citation.ordinal === numeral);
}

/** Evidence is displayed in the same order the answer's markers use. */
export function orderedCitations(citations: Citation[]): Citation[] {
  return [...citations].sort((a, b) => {
    const left = a.ordinal ?? Number.MAX_SAFE_INTEGER;
    const right = b.ordinal ?? Number.MAX_SAFE_INTEGER;
    return left - right;
  });
}

export function paragraphs(text: string): string[] {
  return text
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
}

export function parseParagraph(
  text: string,
  knownNumerals: Set<number>
): InlineSegment[] {
  const segments: InlineSegment[] = [];
  let cursor = 0;

  for (const match of text.matchAll(MARKER)) {
    const numerals = match[1].split(",").map((value) => Number(value.trim()));
    // One unknown number makes the whole bracket ambiguous; leave it alone.
    if (!numerals.every((numeral) => knownNumerals.has(numeral))) continue;

    const start = match.index ?? 0;
    if (start > cursor) {
      segments.push({ kind: "text", text: text.slice(cursor, start) });
    }
    for (const numeral of numerals) {
      segments.push({ kind: "citation", numeral });
    }
    cursor = start + match[0].length;
  }

  if (cursor < text.length) {
    segments.push({ kind: "text", text: text.slice(cursor) });
  }
  return segments;
}

export function knownNumerals(citations: Citation[]): Set<number> {
  return new Set(
    citations
      .map(numeralOf)
      .filter((numeral): numeral is number => numeral !== null)
  );
}

/**
 * A short opening quote from a chunk, cut at a word boundary.
 *
 * A chunk is a retrieval unit -- for a DOCX it can span a dozen paragraphs.
 * Printing it whole buries the two sentences the reader needs to check the
 * answer; the rest belongs behind an explicit "full passage".
 */
export function excerpt(text: string, limit = 240): string {
  const collapsed = text.replace(/\s+/g, " ").trim();
  if (collapsed.length <= limit) return collapsed;

  const cut = collapsed.slice(0, limit);
  const lastSpace = cut.lastIndexOf(" ");
  // A limit this short is a caller error, not a reason to return one word.
  const body = lastSpace > limit * 0.6 ? cut.slice(0, lastSpace) : cut;
  return `${body.trimEnd()}…`;
}
