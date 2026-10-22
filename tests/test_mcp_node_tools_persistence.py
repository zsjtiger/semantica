"""Persistence guards for the graph tools update_node and delete_node.

These tools were ported from the retired ``semantica.mcp_server`` module, which
saved without checking whether the startup load had succeeded. A failed load
then let the next update overwrite the original graph file with an empty one.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from semantica.context import ContextGraph
from semantica.mcp import session
from semantica.mcp.tools.graph import handle_delete_node, handle_update_node


def _graph_with_task() -> ContextGraph:
    graph = ContextGraph(advanced_analytics=False)
    graph.add_node("task-1", "action", label="Write report")
    return graph


class _GraphSession(unittest.TestCase):
    """Install a graph as the session singleton for the duration of a test."""

    def setUp(self):
        self._saved = (session._graph, session._load_ok)
        self.addCleanup(self._restore)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "graph.json")
        self.graph = _graph_with_task()
        self.graph.save_to_file(self.path)
        self._env = patch.dict(os.environ, {"SEMANTICA_KG_PATH": self.path})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _restore(self):
        session._graph, session._load_ok = self._saved

    def _file_nodes(self):
        with open(self.path, encoding="utf-8") as handle:
            return {node["id"] for node in json.load(handle)["nodes"]}


class TestUpdateNodePersistence(_GraphSession):

    def test_updates_and_persists_the_change(self):
        session._graph, session._load_ok = self.graph, True
        result = handle_update_node({"node_id": "task-1", "properties": {"status": "done"}})
        self.assertNotIn("error", result, result)
        self.assertTrue(result["persisted"])
        with open(self.path, encoding="utf-8") as handle:
            self.assertIn("done", handle.read())

    def test_refuses_to_overwrite_a_file_that_failed_to_load(self):
        session._graph, session._load_ok = self.graph, False
        before = open(self.path, encoding="utf-8").read()
        result = handle_update_node({"node_id": "task-1", "properties": {"status": "done"}})
        self.assertIn("Persistence blocked", result.get("error", ""))
        self.assertEqual(open(self.path, encoding="utf-8").read(), before)

    def test_rolls_back_the_in_memory_change_when_the_save_fails(self):
        session._graph, session._load_ok = self.graph, True
        before = dict(self.graph.find_node("task-1")["metadata"])
        with patch.object(self.graph, "save_to_file", side_effect=OSError("disk full")):
            result = handle_update_node({"node_id": "task-1", "properties": {"status": "done"}})
        self.assertIn("Mutation rolled back", result.get("error", ""))
        self.assertEqual(dict(self.graph.find_node("task-1")["metadata"]), before)


class TestDeleteNodePersistence(_GraphSession):

    def test_archives_and_persists_the_node(self):
        session._graph, session._load_ok = self.graph, True
        result = handle_delete_node({"node_id": "task-1"})
        self.assertEqual(result["status"], "archived")
        self.assertTrue(result["persisted"])
        self.assertIn("task-1", self._file_nodes())
        self.assertEqual(self.graph.find_node("task-1")["metadata"]["status"], "archived")

    def test_refuses_to_overwrite_a_file_that_failed_to_load(self):
        session._graph, session._load_ok = self.graph, False
        before = open(self.path, encoding="utf-8").read()
        result = handle_delete_node({"node_id": "task-1"})
        self.assertIn("Persistence blocked", result.get("error", ""))
        self.assertEqual(open(self.path, encoding="utf-8").read(), before)
        self.assertNotEqual(self.graph.find_node("task-1")["metadata"].get("status"), "archived")

    def test_rolls_back_when_the_save_fails(self):
        session._graph, session._load_ok = self.graph, True
        with patch.object(self.graph, "save_to_file", side_effect=OSError("disk full")):
            result = handle_delete_node({"node_id": "task-1"})
        self.assertIn("Mutation rolled back", result.get("error", ""))
        self.assertNotEqual(self.graph.find_node("task-1")["metadata"].get("status"), "archived")


if __name__ == "__main__":
    unittest.main()
