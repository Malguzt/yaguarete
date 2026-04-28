#!/usr/bin/env python3
"""
Script para arreglar comentarios trailing (S139) en archivos Python.
Mueve los comentarios trailing a líneas anteriores.
"""

import os
import re
from pathlib import Path

def fix_trailing_comments(file_path):
    """Arregla comentarios trailing en un archivo"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Detectar línea con comentario trailing
        # Patrón: código + espacio + # comentario (no es un string)
        match = re.match(r'^(\s*)(.*?)\s+(#.*)$', line)
        
        if match and not line.strip().startswith('#'):
            indent = match.group(1)
            code = match.group(2)
            comment = match.group(3)
            
            # Evitar si el código es vacío o si es una cadena literal
            if code and not code.startswith('"""') and not code.startswith("'''"):
                # Agregar línea de código sin comentario
                new_lines.append(f"{indent}{code}\n")
                # Agregar comentario en línea anterior
                new_lines.append(f"{indent}{comment}\n")
                modified = True
                i += 1
                continue
        
        new_lines.append(line)
        i += 1
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        return True
    return False

def main():
    src_dir = Path("src")
    fixed_files = []
    
    for py_file in src_dir.rglob("*.py"):
        if fix_trailing_comments(str(py_file)):
            fixed_files.append(str(py_file))
            print(f"✅ Arreglado: {py_file}")
    
    if fixed_files:
        print(f"\n✓ Se arreglaron {len(fixed_files)} archivos")
    else:
        print("No se encontraron comentarios trailing para arreglar")

if __name__ == "__main__":
    main()
