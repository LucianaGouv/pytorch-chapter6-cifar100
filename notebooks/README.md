# Notebooks

Esta pasta contém os notebooks primários para publicação. O notebook principal é `Cifar100.ipynb`.

Pontos importantes
- Mantenha os notebooks executáveis e auto-contidos sempre que possível (imports, seed, detecção de device).
- Não comite datasets grandes; use flags de download e configure `data/` no `.gitignore`.

Para reexecutar e incorporar as saídas:
```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/Cifar100.ipynb --ExecutePreprocessor.timeout=2400
```
