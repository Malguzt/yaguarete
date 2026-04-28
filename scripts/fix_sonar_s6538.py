#!/usr/bin/env python3
"""
Script para agregar type hints básicos a funciones sin return type.
"""

import os
import re
from pathlib import Path

def add_return_type_hints(file_path):
    """Agrega type hints básicos a funciones"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Detectar definición de función: def nombre(...):
        if re.match(r'^\s*def\s+\w+\s*\(', line):
            # Verificar si ya tiene return type
            if ' -> ' not in line:
                # Analizar qué retorna la función
                # Por ahora, agregar -> None como default si es obvio
                
                # Buscar si retorna algo en los próximos 20 líneas
                has_return = False
                has_explicit_return = False
                for j in range(i + 1, min(i + 30, len(lines))):
                    if re.search(r'\breturn\s+\S', lines[j]):
                        has_explicit_return = True
                        break
                    if lines[j].strip().startswith('def ') or lines[j].strip().startswith('class '):
                        break
                
                # Si es una función que retorna explícitamente, agregar -> Any
                # Si es solo procedimiento (sin return o solo return), agregar -> None
                if has_explicit_return:
                    # Agregar -> Any
                    line = line.rstrip(':\n') + ' -> Any:\n'
                    modified = True
                else:
                    # Agregar -> None
                    line = line.rstrip(':\n') + ' -> None:\n'
                    modified = True
        
        new_lines.append(line)
        i += 1
    
    if modified:
        # Agregar import de typing.Any si es necesario
        content = ''.join(new_lines)
        if ' -> Any' in content and 'from typing import' in content:
            content = re.sub(
                r'from typing import ([^\n]+)',
                lambda m: f"from typing import {m.group(1)}, Any" if 'Any' not in m.group(1) else m.group(0),
                content,
                count=1
            )
        elif ' -> Any' in content:
            # Agregar import
            lines = content.split('\n')
            for idx, l in enumerate(lines):
                if l.startswith('import ') or l.startswith('from '):
                    lines.insert(idx, 'from typing import Any')
                    break
            content = '\n'.join(lines)
        
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    src_dir = Path("src")
    fixed_files = []
    
    for py_file in src_dir.rglob("*.py"):
        if add_return_type_hints(str(py_file)):
            fixed_files.append(str(py_file))
            print(f"✅ Arreglado: {py_file}")
    
    if fixed_files:
        print(f"\n✓ Se agregaron type hints a {len(fixed_files)} archivos")
    else:
        print("No se encontraron funciones sin type hints")

if __name__ == "__main__":
    main()
