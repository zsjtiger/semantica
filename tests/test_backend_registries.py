"""Registry consistency for the vector and graph store backends.

The facades look backends up by name in ``_BACKEND_STORES`` and
``_GRAPH_BACKENDS``. A renamed module or class would only fail when that
backend is actually selected, and most backends need optional services, so
these tests check the tables statically: each entry must name a module that
exists and a class that module defines.
"""

import ast
import pathlib
import unittest

from semantica.graph_store import graph_store
from semantica.vector_store import vector_store

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "semantica"


def _defined_classes(module_path: pathlib.Path) -> set:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


class TestBackendRegistries(unittest.TestCase):

    def _assert_resolves(self, package, module_name, class_name):
        module_path = PACKAGE_ROOT / package / f"{module_name}.py"
        self.assertTrue(module_path.is_file(), f"missing module {module_path}")
        self.assertIn(
            class_name, _defined_classes(module_path),
            f"{module_path.name} does not define {class_name}",
        )

    def test_vector_backends_resolve(self):
        for name, (module_name, class_name, _) in vector_store._BACKEND_STORES.items():
            with self.subTest(backend=name):
                self._assert_resolves("vector_store", module_name, class_name)

    def test_graph_backends_resolve(self):
        for name, (module_name, class_name, getter) in graph_store._GRAPH_BACKENDS.items():
            with self.subTest(backend=name):
                self._assert_resolves("graph_store", module_name, class_name)
                self.assertTrue(
                    callable(getattr(graph_store.graph_store_config, getter, None)),
                    f"graph_store_config has no {getter}",
                )

    def test_every_vector_backend_is_advertised(self):
        self.assertEqual(
            set(vector_store._BACKEND_STORES) | {"inmemory"},
            set(vector_store.VectorStore.SUPPORTED_BACKENDS),
        )


if __name__ == "__main__":
    unittest.main()
