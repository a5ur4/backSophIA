import os
import subprocess
import sys

def run_tests():
    """Executa todos os testes da aplicação."""
    print("Preparando ambiente de teste...")
    
    # Criar diretórios necessários
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Executar testes com pytest
    print("\nExecutando testes...")
    result = subprocess.run(
        ["python", "-m", "pytest", "tests", "-v"],
        capture_output=True,
        text=True
    )
    
    # Exibir resultados
    print(result.stdout)
    if result.stderr:
        print("Erros:")
        print(result.stderr)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())