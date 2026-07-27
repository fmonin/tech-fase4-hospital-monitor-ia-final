<#
PowerShell helper: cria o venv em .venv e instala dependências.
Uso (PowerShell):
  .\scripts\setup_venv.ps1

O script tenta usar o launcher `py -3.11` quando disponível; caso
contrário tenta `python`. Não ativa o venv automaticamente (permissões
de execução podem impedir). Depois rode `.\.venv\Scripts\Activate.ps1`.
#>

Write-Host "Removendo .venv antigo (se existir)..."
if (Test-Path -Path .venv) { Remove-Item -Recurse -Force .venv }

$created = $false
try {
    Write-Host "Tentando criar venv com 'py -3.11'..."
    py -3.11 -m venv .venv
    $created = $true
} catch {
    Write-Host "py launcher não disponível ou Python 3.11 não encontrado, tentando 'python -m venv'..."
    try {
        python -m venv .venv
        $created = $true
    } catch {
        Write-Error "Não foi possível criar o venv automaticamente. Instale Python 3.10/3.11 e certifique-se de que 'py' ou 'python' esteja no PATH."
        exit 1
    }
}

if ($created) {
    Write-Host "Venv criado. Ative-o com: .\.venv\Scripts\Activate.ps1"
    Write-Host "Instalando dependências (demo + treino)."
    .\.venv\Scripts\python -m pip install --upgrade pip
    .\.venv\Scripts\python -m pip install -r requirements.txt
    Write-Host "Pronto. Se quiser as dependências opcionais (MediaPipe/YAMNet), rode: .\.venv\Scripts\python -m pip install -r requirements-optional.txt"
}
