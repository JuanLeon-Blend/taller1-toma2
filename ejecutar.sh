#!/bin/bash
set -e

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno
source venv/bin/activate

# Instalar dependencias
echo "Instalando dependencias..."
pip install -q -r requirements.txt

# Ejecutar notebook y generar versión con outputs
echo "Ejecutando notebook..."
jupyter nbconvert --to notebook --execute informe.ipynb \
    --output informe_ejecutado.ipynb \
    --ExecutePreprocessor.timeout=120

# Extraer gráficas a img/
echo "Extrayendo gráficas..."
python3 -c "
import json, base64, os
os.makedirs('img', exist_ok=True)
with open('informe_ejecutado.ipynb') as f:
    nb = json.load(f)
n = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'display_data' and 'image/png' in out.get('data', {}):
                n += 1
                with open(f'img/grafica_{n}.png', 'wb') as img:
                    img.write(base64.b64decode(out['data']['image/png']))
print(f'{n} gráficas extraídas en img/')
"

echo "Listo. Archivos generados:"
echo "  - informe_ejecutado.ipynb (notebook con outputs)"
echo "  - img/ (gráficas PNG)"
echo "  - informe_hallazgos.md (informe final)"
