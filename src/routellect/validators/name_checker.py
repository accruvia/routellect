"""AST-based undefined name checker for generated code.

Walks the AST to find names that are used (loaded) but never defined
or imported. Catches missing imports, typos, and hallucinated symbols
before code reaches pytest.

Fast (AST-only, no execution) and general-purpose — not tied to any
specific library or import source.
"""

import ast
import builtins
from dataclasses import dataclass, field

# All Python builtins available without import
_BUILTINS = frozenset(dir(builtins))


@dataclass
class UndefinedName:
    """A name used in code that has no visible definition."""

    name: str
    lineno: int
    col_offset: int

    def __str__(self) -> str:
        return f"'{self.name}' (line {self.lineno})"


class NameChecker:
    """Finds undefined names in Python source code.

    Walks the AST and collects all names that are loaded (used) but
    never defined via import, assignment, function/class def, or
    loop/comprehension target.
    """

    def check(self, code: str) -> list[UndefinedName]:
        """Return list of undefined names in code. Empty = all good."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []  # Syntax errors are caught elsewhere

        defined = set(_BUILTINS)
        self._collect_definitions(tree, defined)
        return self._find_undefined(tree, defined)

    def _collect_definitions(self, tree: ast.AST, defined: set[str]) -> None:
        """Collect all names that are defined in the module scope."""
        for node in ast.walk(tree):
            # Imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)

            # Assignments
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in self._assignment_targets(node):
                    defined.add(target)

            # Function and class definitions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
                # Function arguments are defined inside
                for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                    defined.add(arg.arg)
                if node.args.vararg:
                    defined.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined.add(node.args.kwarg.arg)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)

            # For-loop and comprehension targets
            elif isinstance(node, ast.For):
                self._collect_target_names(node.target, defined)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for gen in node.generators:
                    self._collect_target_names(gen.target, defined)

            # With-statement targets
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars:
                        self._collect_target_names(item.optional_vars, defined)

            # Exception handler variable
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    defined.add(node.name)

            # Walrus operator
            elif isinstance(node, ast.NamedExpr):
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)

            # Global / nonlocal (these reference names that exist elsewhere)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                for name in node.names:
                    defined.add(name)

    def _assignment_targets(self, node: ast.AST) -> list[str]:
        """Extract target names from assignment nodes."""
        names = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                self._collect_target_names(target, names)
        elif isinstance(node, ast.AnnAssign) and node.target:
            self._collect_target_names(node.target, names)
        return names

    def _collect_target_names(
        self, node: ast.AST, collector: set | list
    ) -> None:
        """Recursively collect names from assignment/loop targets."""
        add = collector.add if isinstance(collector, set) else collector.append
        if isinstance(node, ast.Name):
            add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._collect_target_names(elt, collector)
        elif isinstance(node, ast.Starred):
            self._collect_target_names(node.value, collector)

    def _find_undefined(
        self, tree: ast.AST, defined: set[str]
    ) -> list[UndefinedName]:
        """Find all Name nodes that load an undefined name."""
        undefined = []
        seen = set()

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in defined
                and node.id not in seen
            ):
                seen.add(node.id)
                undefined.append(
                    UndefinedName(
                        name=node.id,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                    )
                )

        return undefined
