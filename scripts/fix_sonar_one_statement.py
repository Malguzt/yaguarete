#!/usr/bin/env python3
"""
Script para separar múltiples statements por línea (OneStatementPerLine)
"""

import os
import re
from pathlib import Path

def fix_one_statement_per_line(file_path):
    """Separa múltiples statements en una sola línea"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for line in lines:
        # Detectar patrones: if/elif/else/for/while + statement en la misma línea
        # Ejemplos: if x: y = 1  -->  if x:\n    y = 1
        
        # Pattern: if/elif/else/for/while/try condition: statement
        match = re.match(r'^(\s*)(if|elif|else|for|while|try|except|finally|with)\s+(.+?):\s+(\S.*)$', line)
        
        if match:
            indent = match.group(1)
            keyword = match.group(2)
            condition = match.group(3) if match.group(2) != 'else' else ''
            statement = match.group(4)
            
            # Asegurarse de que no sea un comentario
            if not statement.startswith('#'):
                # Separar la condición del statement
                if keyword == 'else':
                    new_lines.append(f"{indent}{keyword}:\n")
                else:
                    new_lines.append(f"{indent}{keyword} {condition}:\n")
                new_lines.append(f"{indent}    {statement}\n")
                modified = True
                continue
        
        new_lines.append(line)
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        return True
    return False

def main():
    src_dir = Path("src")
    fixed_files = []
    
    for py_file in src_dir.rglob("*.py"):
        if fix_one_statement_per_line(str(py_file)):
            fixed_files.append(str(py_file))
            print(f"✅ Arreglado: {py_file}")
    
    if fixed_files:
        print(f"\n✓ Se separaron statements en {len(fixed_files)} archivos")
    else:
        print("No se encontraron múltiples statements por línea")

if __name__ == "__main__":
    main()
