# Sampai MPI - Guide d'Utilisation

## Compilation Séparée

Sampai nécessite une compilation SEPARÉE pour MPI et non-MPI :

```bash
# Sans MPI (version série)
pip install .

# Avec MPI (version parallèle)
pip install . --global-option="-Dmpi=true"
# Ou avec meson directement:
meson setup builddir -Dmpi=true
meson compile -C builddir
```

## Utilisation

### 1. Version SÉRIE (sans MPI)
```bash
# Compilation
pip install .

# Exécution
python demo.py
```

### 2. Version PARALLÈLE (avec MPI)
```bash
# Compilation
pip install . --global-option="-Dmpi=true"

# Exécution avec 4 rangs
mpiexec -n 4 python demo.py

# Exécution avec 1 rang (marche aussi!)
mpiexec -n 1 python demo.py

# Exécution avec 8 rangs
mpiexec -n 8 python demo.py
```

## API Python

Le MÊME code Python fonctionne pour les deux cas :

```python
from mpi4py import MPI  # Importé en premier pour MPI
import sampai as sam

# Vérifier si MPI est disponible
try:
    from sampai import mpi
    has_mpi = mpi.is_initialized()
except (ImportError, RuntimeError):
    has_mpi = False

# Créer un mesh (auto-distribué si MPI)
box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
config = sam.config.make(2)
mesh = sam.mesh.make(box, config)

# Créer un champ
u = sam.field.zeros(mesh, "u")

# Ghosts (auto-MPI)
sam.adaptation.update_ghost_mr(u)

# Save (auto-MPI)
sam.save("output.h5", u)
```

## Module `sampai.mpi`

Uniquement disponible si compilé avec `-Dmpi=true` :

```python
from sampai import mpi

# Fonctions disponibles
- mpi.rank()              # Rang du processus (0-indexed)
- mpi.size()              # Nombre total de processus
- mpi.barrier()           # Synchronisation
- mpi.is_initialized()    # True si MPI activé
- mpi.Communicator        # Classe communicateur
- mpi.init()              # Initialiser MPI
- mpi.finalize()          # Finaliser MPI (no-op)
```

## Comportement automatique Samurai

| Fonction | Sans MPI | Avec MPI |
|----------|----------|----------|
| `sam.mesh.make()` | Mesh complet | Mesh distribué (auto) |
| `sam.adaptation.update_ghost_mr()` | Ghosts locaux | Ghosts + échange MPI (auto) |
| `sam.save("f.h5", u)` | `f.h5` | `f_size_4.h5` (auto) |
| `sam.load("f.h5")` | Charge série | Charge parallèle (auto) |

## Exemples

Voir `demo_mpi_example.py` pour un exemple complet qui fonctionne avec et sans MPI.
