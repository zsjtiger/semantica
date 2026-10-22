"""Row-to-document conversion shared by the database ingestors.

Snowflake, Databricks and Redshift all turn result rows into the same
``{"id", "metadata", "text"}`` shape. Only the ``source`` label and the table
location fields differ, so each ingestor passes those in as ``context``.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence


def rows_to_documents(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    context: Mapping[str, Any],
    id_field: str = "id",
    text_fields: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert rows to document dictionaries.

    The document ``id`` is ``str(row[id_field])``, falling back to the row index
    when the field is absent. When *text_fields* is given, the text is those
    fields' non-``None`` values joined by spaces. Otherwise it is every
    ``str``-typed value in the row.
    """
    documents: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        doc: Dict[str, Any] = {
            "id": str(row.get(id_field, idx)),
            "metadata": {"source": source, **context},
        }
        if text_fields:
            text_parts = [str(row[f]) for f in text_fields if f in row and row[f] is not None]
        else:
            text_parts = [value for value in row.values() if isinstance(value, str)]
        doc["text"] = " ".join(text_parts)
        doc["metadata"]["row_data"] = row
        documents.append(doc)
    return documents
