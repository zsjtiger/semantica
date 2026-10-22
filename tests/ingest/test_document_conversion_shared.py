"""Snowflake, Databricks and Redshift share one row-to-document conversion.

The expected values below were derived from each ingestor's conversion logic
before it moved to ``rows_to_documents``, so these tests check that the shared
helper did not change any output.
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

from semantica.ingest._documents import rows_to_documents
from semantica.ingest.databricks_ingestor import DatabricksData, DatabricksIngestor
from semantica.ingest.redshift_ingestor import RedshiftData, RedshiftIngestor

ROWS = [
    {"id": 7, "name": "Ada", "notes": "first", "score": 3.5, "active": True, "extra": None},
    {"name": "Grace", "notes": None, "score": 9},
]


def _snowflake_classes():
    """Import the Snowflake ingestor with the driver stubbed, as its own tests do.

    The import is scoped so the module is dropped from sys.modules afterwards;
    a cached copy would leak into the Snowflake tests that run later.
    """
    with patch.dict("sys.modules", {"snowflake": MagicMock(), "snowflake.connector": MagicMock()}):
        from semantica.ingest.snowflake_ingestor import SnowflakeData, SnowflakeIngestor
    return SnowflakeData, SnowflakeIngestor


class _Stub:
    """Stands in for an ingestor instance: the conversion only needs a logger."""

    logger = logging.getLogger(__name__)


class TestRowsToDocuments(unittest.TestCase):

    def test_text_fields_join_non_null_values(self):
        docs = rows_to_documents(
            ROWS, source="s", context={"table": "t"}, text_fields=["name", "notes"]
        )
        self.assertEqual(docs[0]["text"], "Ada first")
        self.assertEqual(docs[1]["text"], "Grace")

    def test_without_text_fields_only_string_values_are_used(self):
        docs = rows_to_documents(ROWS, source="s", context={})
        self.assertEqual(docs[0]["text"], "Ada first")
        self.assertEqual(docs[1]["text"], "Grace")

    def test_id_falls_back_to_row_index(self):
        docs = rows_to_documents(ROWS, source="s", context={})
        self.assertEqual(docs[0]["id"], "7")
        self.assertEqual(docs[1]["id"], "1")

    def test_metadata_carries_source_context_and_row(self):
        docs = rows_to_documents(ROWS, source="snowflake", context={"table": "CUSTOMERS", "schema": "PUBLIC"})
        self.assertEqual(
            docs[0]["metadata"],
            {"source": "snowflake", "table": "CUSTOMERS", "schema": "PUBLIC", "row_data": ROWS[0]},
        )

    def test_key_order_matches_the_documented_shape(self):
        docs = rows_to_documents(ROWS, source="s", context={})
        self.assertEqual(list(docs[0]), ["id", "metadata", "text"])


class TestSharedConversionPerIngestor(unittest.TestCase):

    def test_snowflake(self):
        SnowflakeData, SnowflakeIngestor = _snowflake_classes()
        data = SnowflakeData(
            data=ROWS, row_count=2, columns=["id", "name"],
            table_name="CUSTOMERS", database="DB", schema="PUBLIC",
        )
        docs = SnowflakeIngestor.export_as_documents(_Stub(), data, text_fields=["name"])
        self.assertEqual(docs[0]["metadata"]["source"], "snowflake")
        self.assertEqual(docs[0]["metadata"]["database"], "DB")
        self.assertEqual(docs[0]["text"], "Ada")
        self.assertEqual(docs[1]["id"], "1")

    def test_databricks_uses_catalog(self):
        data = DatabricksData(
            data=ROWS, row_count=2, columns=["id", "name"],
            table_name="customers", catalog="main", schema="sales",
        )
        docs = DatabricksIngestor.export_as_documents(_Stub(), data)
        self.assertEqual(
            {k: v for k, v in docs[0]["metadata"].items() if k != "row_data"},
            {"source": "databricks", "table": "customers", "catalog": "main", "schema": "sales"},
        )
        self.assertEqual(docs[0]["text"], "Ada first")

    def test_redshift(self):
        data = RedshiftData(
            data=ROWS, row_count=2, columns=["id", "name"],
            table_name="orders", database="dev", schema="public",
        )
        docs = RedshiftIngestor.export_as_documents(_Stub(), data, id_field="name")
        self.assertEqual(docs[0]["id"], "Ada")
        self.assertEqual(docs[0]["metadata"]["source"], "redshift")
        self.assertEqual(docs[0]["metadata"]["row_data"], ROWS[0])


if __name__ == "__main__":
    unittest.main()
