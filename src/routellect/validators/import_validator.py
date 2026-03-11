"""Validates that imports in generated code are resolvable."""

import ast
import importlib
import importlib.util
import sys
from pathlib import Path


class ImportValidator:
    """Validates imports in generated test code."""

    def __init__(self, src_root: Path = None):
        self.src_root = src_root or Path("src")
        # Add src to path for validation
        src_str = str(self.src_root.absolute())
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    def validate(self, code: str, allowlist: set[str] | None = None) -> tuple[bool, list[str]]:
        """Check if all imports in code are resolvable.

        Args:
            code: Python code to validate
            allowlist: Package names or dotted module paths to skip validation
                for. Matches by top-level package (e.g. "slowapi" skips
                "import slowapi") or by module prefix (e.g.
                "accruvia.models.task" skips "from accruvia.models.task
                import Foo").

        Returns:
            Tuple of (all_valid, list_of_invalid_imports)
        """
        allowlist = allowlist or set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False, ["<syntax error>"]

        invalid = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if self._is_allowed(alias.name, allowlist):
                        continue
                    if not self._can_import(alias.name):
                        invalid.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                # Skip relative imports (from .foo import bar) — can't validate
                # without knowing the file's position in the package hierarchy
                if node.level and node.level > 0:
                    continue
                if node.module:
                    if self._is_allowed(node.module, allowlist):
                        continue
                    # Check if the module exists
                    if not self._can_import(node.module):
                        invalid.append(node.module)
                    else:
                        # Check if the specific names exist in the module
                        for alias in node.names:
                            if alias.name != "*":
                                full_name = f"{node.module}.{alias.name}"
                                if not self._can_import_from(node.module, alias.name):
                                    invalid.append(full_name)

        return len(invalid) == 0, invalid

    @staticmethod
    def _is_allowed(module_name: str, allowlist: set[str]) -> bool:
        """Check if a module matches any allowlist entry.

        Matches if the top-level package is in the allowlist (e.g. "slowapi")
        or if the module starts with a dotted allowlist entry (e.g.
        "accruvia.models.task" allows "accruvia.models.task.sub").
        """
        top = module_name.split(".")[0]
        if top in allowlist:
            return True
        return any(module_name == entry or module_name.startswith(entry + ".") for entry in allowlist if "." in entry)

    def _can_import(self, module_name: str) -> bool:
        """Check if a module can be imported."""
        try:
            # First check if it's already loaded
            if module_name in sys.modules:
                return True
            # Try to find the spec without actually importing
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                return True
        except (ModuleNotFoundError, ImportError, ValueError):
            pass

        # Fallback: check if the .py file exists on disk under src_root.
        # Handles newly-created modules in worktrees that aren't yet
        # importable via find_spec (cached parent package).
        return self._file_exists_for_module(module_name)

    def _file_exists_for_module(self, module_name: str) -> bool:
        """Check if a .py file or package exists for this module under src_root."""
        parts = module_name.split(".")
        base = self.src_root
        for part in parts:
            candidate = base / part
            if (candidate.with_suffix(".py")).exists():
                return True
            if candidate.is_dir() and (candidate / "__init__.py").exists():
                base = candidate
                continue
            return False
        return True  # All parts resolved as packages

    def _can_import_from(self, module_name: str, attr_name: str) -> bool:
        """Check if an attribute can be imported from a module."""
        try:
            # For accruvia modules, actually import to check
            if module_name.startswith("accruvia"):
                mod = importlib.import_module(module_name)
                return hasattr(mod, attr_name)
            # For external modules, just check the module exists
            return self._can_import(module_name)
        except (ModuleNotFoundError, ImportError, AttributeError):
            # If the module file exists on disk but can't be imported
            # (e.g. newly created in a worktree), trust the file exists
            if self._file_exists_for_module(module_name):
                return True
            return False

    def get_import_feedback(self, invalid_imports: list[str]) -> str:
        """Generate feedback message for invalid imports."""
        if not invalid_imports:
            return ""

        imports_list = ", ".join(invalid_imports[:5])
        return f"These imports don't exist: {imports_list}. Use only the APIs shown in the context."
