"""
Demo Sampai : Utilisation avec/sans MPI

Ce code fonctionne EN:
- Série (sans MPI) : python demo_mpi_example.py
- Parallèle (avec MPI) : mpiexec -n 4 python demo_mpi_example.py

NOTE: Si compilé avec MPI, vous pouvez utiliser mpiexec avec n'importe quel nombre de rangs.
"""

from mpi4py import MPI  # Nécessaire pour MPI - importé en premier
import sampai as sam

# Le module mpi est disponible SEULEMENT si compilé avec -Dmpi=true
try:
    from sampai import mpi
    HAS_MPI = mpi.is_initialized()
except (ImportError, RuntimeError):
    HAS_MPI = False
    mpi = None


def main():
    # Afficher le mode d'exécution
    if HAS_MPI:
        if mpi.rank() == 0:
            print(f"=== Mode MPI avec {mpi.size()} processus ===")
        print(f"Rank {mpi.rank()}/{mpi.size()}: démarrage")
    else:
        print("=== Mode SÉRIE (sans MPI) ===")

    # Paramètres de simulation
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.config.make(2)
    config.min_level = 2
    config.max_level = 3

    # Créer le mesh (auto-distribué si MPI activé)
    mesh = sam.mesh.make(box, config)

    # Afficher les cellules par rang
    if HAS_MPI:
        print(f"Rank {mpi.rank()}: {mesh.nb_cells} cellules", flush=True)
    else:
        print(f"Cellules totales: {mesh.nb_cells}")

    # Créer un champ
    u = sam.field.zeros(mesh, "u")

    # Mettre à jour les ghosts (auto-MPI si activé)
    sam.adaptation.update_ghost_mr(u)

    # Sauvegarder (auto-MPI si activé)
    # Note: En MPI, Samurai ajoute automatiquement _size_N au nom de fichier
    sam.save("demo_output.h5", u)

    if HAS_MPI:
        mpi.barrier()
        if mpi.rank() == 0:
            print("Sauvegarde terminée: demo_output.h5 (avec suffixe _size_N en MPI)")
    else:
        print("Sauvegarde terminée: demo_output.h5")


if __name__ == "__main__":
    main()
