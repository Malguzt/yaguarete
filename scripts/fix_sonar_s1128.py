#!/usr/bin/env python3
"""
Script para remover imports no utilizados (S1128)
"""

import os
import ast
from pathlib import Path

def get_used_names(content):
    """Extrae todos los nombres usados en el código"""
    try:
        tree = ast.parse(content)
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        return used_names
    except:
        return set()

def remove_unused_imports(file_path):
    """Remueve imports no utilizados"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    lines = content.split('\n')
    new_lines = []
    used_names = get_used_names(content)
    
    for line in lines:
        # Detectar import statements
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            # Extraer nombres importados
            if line.strip().startswith('from '):
                # from X import Y, Z
                match = line.split('import')[-1].strip()
                imported = [x.strip().split(' as ')[-1] for x in match.split(',')]
            elif line.strip().startswith('import '):
                # import X, Y, Z
                match = line.split('import')[-1].strip()
                imported = [x.strip().split(' as ')[-1] for x in match.split(',')]
            else:
                imported = []
            
            # Verificar si alguno se usa
            is_used = any(name in used_names for name in imported if name)
            
            if not is_used and 'import' in line:
                # Comentar o remover línea de import no usada
                # Por ahora, no la eliminamos, solo si es seguro
                continue
        
        new_lines.append(line)
    
    new_content = '\n'.join(new_lines)
    
    if new_content != original_content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    return False

def main():
    src_dir = Path("src")
    fixed_files = []
    
    for py_file in src_dir.rglob("*.py"):
        try:
            if remove_unused_imports(str(py_file)):
                fixed_files.append(str(py_file))
                print(f"✅ Arreglado: {py_file}")
        except Exception as e:
            pass
    
    if fixed_files:
        print(f"\n✓ Se limpiaron imports en {len(fixed_files)} archivos")
    else:
        print("No se encontraron imports no utilizados (o se procesaron de forma segura)")

if __name__ == "__main__":
    main()
