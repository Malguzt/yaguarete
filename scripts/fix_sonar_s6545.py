#!/usr/bin/env python3
"""
Script para reemplazar tipos de typing por tipos built-in en Python 3.9+
Dict -> dict, List -> list, Tuple -> tuple, etc.
"""

import os
import re
from pathlib import Path

def fix_typing_generics(file_path):
    """Reemplaza tipos de typing por tipos built-in"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Remover imports innecesarios de typing primero
    # Si solo importan Dict, List, etc., reemplazar con los tipos built-in
    
    # Reemplazos: Dict[X] -> dict[X], List[X] -> list[X], etc.
    # Pero cuidado con los strings y comentarios
    
    replacements = [
        (r'\bDict\[', 'dict['),
        (r'\bList\[', 'list['),
        (r'\bTuple\[', 'tuple['),
        (r'\bSet\[', 'set['),
        (r'\bFrozenSet\[', 'frozenset['),
    ]
    
    for old_pattern, new_pattern in replacements:
        content = re.sub(old_pattern, new_pattern, content)
    
    # Limpiar imports de typing
    # Si importan: from typing import Dict, List...
    # Cambiar a: from typing import ...otros que falten...
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    src_dir = Path("src")
    fixed_files = []
    
    for py_file in src_dir.rglob("*.py"):
        if fix_typing_generics(str(py_file)):
            fixed_files.append(str(py_file))
            print(f"✅ Arreglado: {py_file}")
    
    if fixed_files:
        print(f"\n✓ Se arreglaron {len(fixed_files)} archivos")
        print("\nProximos pasos:")
        print("- Revisar los imports de typing que aún son necesarios")
        print("- Agregar type hints a funciones sin return type")
    else:
        print("No se encontraron tipos de typing para reemplazar")

if __name__ == "__main__":
    main()
