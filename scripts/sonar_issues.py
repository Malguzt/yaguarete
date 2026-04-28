import requests
import sys
import argparse
from collections import Counter
import os
from dotenv import load_dotenv

load_dotenv()

SONAR_URL = os.getenv("SONAR_URL", "http://localhost:9000")
if "sonarqube-local" in SONAR_URL:
    # When running locally (not in Docker), replace service name with localhost
    SONAR_URL = SONAR_URL.replace("sonarqube-local", "localhost")
PROJECT_KEY = "yaguarete"
TOKEN = os.getenv("SONAR_TOKEN", "")

if not TOKEN:
    print("Error: SONAR_TOKEN no está configurado. Verifica tu archivo .env")
    sys.exit(1)

def get_session():
    session = requests.Session()
    session.auth = (TOKEN, "")
    return session

def list_issues():
    session = get_session()
    url = f"{SONAR_URL}/api/issues/search"
    params = {
        "componentKeys": PROJECT_KEY,
        "ps": 100, # Page size
        "resolved": "false"
    }
    
    try:
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        issues = data.get("issues", [])
        if not issues:
            print(f"No se encontraron issues abiertos para el proyecto '{PROJECT_KEY}'.")
            return

        # Count occurrences by message/rule
        counts = Counter([f"{i.get('message')} [{i.get('rule')}]" for i in issues])
        
        print(f"\n--- Resumen de Issues en {PROJECT_KEY} ---")
        print(f"{'Ocurrencias':<12} | {'Issue [Regla]'}")
        print("-" * 50)
        for issue, count in counts.most_common():
            print(f"{count:<12} | {issue}")
            
        print(f"\nTotal: {len(issues)} issues únicos.")
        print("-" * 50)
        print("Usa 'make sonar-issue-detail KEY=<key>' para ver más detalles.")
        print("Puedes obtener las claves (keys) listando todos los issues con el script directamente.")

    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con SonarQube: {e}")
        sys.exit(1)

def list_all_keys():
    session = get_session()
    url = f"{SONAR_URL}/api/issues/search"
    params = {"componentKeys": PROJECT_KEY, "resolved": "false"}
    
    try:
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"\n--- Lista de Claves de Issues (Keys) ---")
        for i in data.get("issues", []):
            print(f"{i.get('key')} -> {i.get('message')}")
    except Exception as e:
        print(f"Error: {e}")

def get_issue_detail(issue_key):
    session = get_session()
    url = f"{SONAR_URL}/api/issues/search"
    params = {"issues": issue_key}
    
    try:
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        issues = data.get("issues", [])
        if not issues:
            print(f"No se encontró el issue con clave: {issue_key}")
            return
            
        issue = issues[0]
        print(f"\n--- Detalle del Issue: {issue_key} ---")
        print(f"Mensaje:    {issue.get('message')}")
        print(f"Regla:      {issue.get('rule')}")
        print(f"Severidad:  {issue.get('severity')}")
        print(f"Componente: {issue.get('component')}")
        print(f"Línea:      {issue.get('line')}")
        print(f"Estado:     {issue.get('status')}")
        print(f"Autor:      {issue.get('author', 'N/A')}")
        print(f"Creación:   {issue.get('creationDate')}")
        print("-" * 50)

    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con SonarQube: {e}")
        sys.exit(1)

def list_top_issues(top_n):
    """Muestra los TOP N issues por ocurrencias y sus claves"""
    session = get_session()
    url = f"{SONAR_URL}/api/issues/search"
    params = {
        "componentKeys": PROJECT_KEY,
        "ps": 100,  # Page size
        "resolved": "false"
    }
    
    try:
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        issues = data.get("issues", [])
        if not issues:
            print(f"No se encontraron issues abiertos para el proyecto '{PROJECT_KEY}'.")
            return

        # Agrupar por message y contar
        issue_groups = {}
        for issue in issues:
            message = f"{issue.get('message')} [{issue.get('rule')}]"
            if message not in issue_groups:
                issue_groups[message] = []
            issue_groups[message].append(issue)
        
        # Ordenar por cantidad de ocurrencias
        sorted_issues = sorted(issue_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        print(f"\n--- Top {top_n} Issues con Mayor Ocurrencias ---")
        for idx, (message, issues_list) in enumerate(sorted_issues[:top_n], 1):
            print(f"\n{idx}. {message} ({len(issues_list)} ocurrencias)")
            print("   Keys:")
            for issue in issues_list:
                print(f"   - {issue.get('key')}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con SonarQube: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Herramienta CLI para SonarQube Issues")
    parser.add_argument("--list", action="store_true", help="Lista resumen de issues")
    parser.add_argument("--keys", action="store_true", help="Lista todas las claves de issues")
    parser.add_argument("--top", type=int, default=0, help="Muestra solo los TOP N issues por ocurrencias y sus keys")
    parser.add_argument("--detail", type=str, help="Muestra el detalle de un issue por su clave (key)")
    
    args = parser.parse_args()
    
    if args.list:
        list_issues()
    elif args.top:
        list_top_issues(args.top)
    elif args.keys:
        list_all_keys()
    elif args.detail:
        get_issue_detail(args.detail)
    else:
        parser.print_help()
