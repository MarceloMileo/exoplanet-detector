# exoplanet-detector

Detecção de exoplanetas pelo **método de trânsito**: quando um planeta passa na
frente da sua estrela, o brilho dela cai um pouco, de forma periódica. Este
projeto procura essas quedas em curvas de luz das missões Kepler e TESS.

## Pipeline

```
curva de luz → pré-processamento → busca BLS → features → classificador → candidato / falso positivo
```

| Etapa | Módulo | Status |
|---|---|---|
| Estrutura da curva de luz | `lightcurve.py` | ✅ |
| Limpeza, normalização e detrending | `preprocess.py` | ✅ |
| Busca de trânsitos (Box Least Squares) | `search.py` | ✅ |
| Curvas sintéticas para testes | `synthetic.py` | ✅ |
| Features + classificador | — | 🔜 |
| Download via lightkurve + CLI | — | 🔜 |

## Uso rápido

```python
from exoplanet_detector import preprocess, search_transit
from exoplanet_detector.synthetic import make_lightcurve

lc = make_lightcurve(period=3.5, depth=0.005)  # estrela simulada com um planeta
candidate = search_transit(preprocess(lc))
print(candidate)  # period≈3.5, depth≈0.005, ...
```

## Desenvolvimento

Usamos [uv](https://docs.astral.sh/uv/) para gerenciar o ambiente:

```bash
uv sync                 # cria o ambiente e instala as dependências
uv run pytest           # testes
uv run ruff check .     # lint
uv run ruff format .    # formatação
```
