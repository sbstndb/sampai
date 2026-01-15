# Simulation Factory Design Document

## Overview

High-level `Simulation` class to orchestrate time-stepping loops with mesh adaptation, eliminating boilerplate across examples.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Implementation** | Python pur | Prototypage rapide, compose existing components, C++ optimization later if needed |
| **API Style** | Builder + `build()` | Fluent declarative API with deferred validation |
| **Predefined Examples** | None | Builder only, examples serve as documentation |
| **Extensibility** | RHS callable + Scheme subclass | Simple (lambda/func) for common cases, class-based for advanced |
| **Hooks System** | Decorators `@hook` | Elegant Pythonic syntax, clear registration |
| **Field Management** | Auto-create | Solution field created by Simulation, temps auto-allocated |
| **Output/Visualization** | Builder `.output()` + `@on_output` hook | Declarative for standards, flexible for custom |
| **Mesh Adaptation** | User-controlled | `every`, `every_n(N)`, or `when(condition)` |
| **Progress Bar** | Auto via `progress.time_loop` | Integrated by default, opt-out if needed |

---

## API Reference

### 1. Builder API (Primary)

```python
import sampai as sam

sim = (
    sam.SimulationBuilder(mesh)
    .scheme('rk3', cfl=0.95)
    .solution('u', init='hat')
    .time(tf=1.0, dt=None)
    .adapt(epsilon=2e-4, frequency='every')
    .output('./results', interval=0.1, format='hdf5')
    .progress(desc='Burgers 2D')
    .build()
)
```

#### Builder Methods

| Method | Parameters | Description |
|--------|------------|-------------|
| `.scheme()` | `name: str` ('euler', 'rk3') or `Scheme` instance, `**kwargs` | Time-stepping scheme with optional params (cfl, rhs) |
| `.solution()` | `name: str`, `init: float|str|callable` | Main solution field, auto-created |
| `.time()` | `tf: float`, `dt: float=None`, `cfl: float` | Time configuration |
| `.adapt()` | `epsilon: float`, `frequency: str|int|callable` | MRA config, 'every', N, or condition(t,iter) |
| `.output()` | `path: str`, `interval: float`, `format: str` | Auto-output configuration |
| `.progress()` | `desc: str`, `enable: bool=True` | Progress bar control |
| `.build()` | - | Creates and validates the `Simulation` |

---

### 2. Decorator Hooks

```python
@sim.before_step
def check_cfl(u, t, iteration):
    assert u.max() < 1e3

@sim.after_step
def update_viz(u, t, iteration):
    viz.update(u)

@sim.before_adapt
def log_mesh(u):
    print(f"Before adapt: {u.mesh.n_cells} cells")

@sim.after_adapt
def log_mesh(u, mesh_stats):
    print(f"After adapt: {mesh_stats.n_cells} cells")

@sim.on_output
def save_custom(u, t, iteration):
    np.save(f'stats_{iteration}.npy', u.max())
```

| Hook | Signature | When Called |
|------|-----------|-------------|
| `@before_step` | `func(u, t, iteration)` | Before time step |
| `@after_step` | `func(u, t, iteration)` | After time step |
| `@before_adapt` | `func(u)` | Before mesh adaptation |
| `@after_adapt` | `func(u, mesh_stats)` | After mesh adaptation |
| `@on_output` | `func(u, t, iteration)` | When output is saved |

---

### 3. Execution Methods

```python
# Full auto-run
u_final = sim.run()

# Step-by-step control
with sim:
    while sim.running:
        u = sim.step()
        # custom logic between steps
```

| Method | Returns | Description |
|--------|---------|-------------|
| `.run()` | `Field` | Runs complete simulation, returns final solution |
| `.step()` | `Field` | Executes single time step with AMR |
| `__enter__()` | `Simulation` | Context manager entry |
| `__exit__()` | - | Context manager exit |

---

### 4. Custom Schemes (Extensibility)

#### Option A: RHS Callable (Simple)

```python
def my_rhs(u):
    return sam.operators.upwind(u, [1.0, 0.5])

sim = sam.SimulationBuilder(mesh).scheme('euler', rhs=my_rhs).build()
```

#### Option B: Scheme Subclass (Advanced)

```python
from sampai.simulation import Scheme

class MyScheme(Scheme):
    def __init__(self, param):
        self.param = param

    def step(self, u, dt) -> Field:
        # Custom time-stepping logic
        return u_next

sim = sam.SimulationBuilder(mesh).scheme(MyScheme(param=1.5)).build()
```

---

## Usage Examples

### Before (Current API - 300+ lines)

```python
# examples/burgers.py - abbreviated
mesh = sam.mesh.make(box, config)
u = sam.field.zeros_vector(mesh, "u", n_components=2)
u1 = sam.field.zeros_vector(mesh, "u1", n_components=2)
u2 = sam.field.zeros_vector(mesh, "u2", n_components=2)
unp1 = sam.field.zeros_vector(mesh, "unp1", n_components=2)
# ... IC, BC, viz setup (150+ lines) ...

with progress.time_loop(Tf, dt, desc="Burgers 2D") as pbar:
    while True:
        with progress.mesh_adaptation(mesh):
            MRadaptation(mra_config)
        u1.resize(); u2.resize(); unp1.resize()
        sam.adaptation.update_ghost_mr(u)
        pbar.advance_time(dt)
        pbar.update_stats(mesh=mesh)
        if not pbar.continue_loop():
            break
        # RK3 stages (15 lines) ...
        # Swap
```

### After (New API - ~30 lines)

```python
import sampai as sam

mesh = sam.mesh.make(box, config)

sim = (
    sam.SimulationBuilder(mesh)
    .scheme('rk3', rhs=lambda u: sam.operators.convection_weno5(u))
    .solution('u', init='hat')
    .time(tf=1.0, cfl=0.95)
    .adapt(epsilon=2e-4)
    .output('./results', interval=0.02)
    .build()
)

u_final = sim.run()
```

---

## File Structure

```
src/sampai/
├── simulation/
│   ├── __init__.py          # Public API exports
│   ├── builder.py           # SimulationBuilder class
│   ├── simulation.py        # Simulation main class
│   ├── timestepper.py       # EulerStepper, RK3Stepper, Scheme ABC
│   ├── field_manager.py     # Auto field allocation
│   └── callbacks.py         # CallbackRegistry, decorator support
└── __init__.py              # Updated: expose sam.SimulationBuilder
```

---

## Supported Schemes

| Scheme | Description | Temp Fields | RHS Required |
|--------|-------------|-------------|--------------|
| `'euler'` | Forward Euler | 1 (`unp1`) | Yes |
| `'rk3'` | TVD-RK3 | 3 (`u1`, `u2`, `unp1`) | Yes |

## Default Behavior

- **Mesh adaptation**: Every time step (configurable)
- **Progress bar**: Auto-enabled (uses `progress.time_loop`)
- **Field allocation**: Automatic based on scheme
- **Ghost update**: Automatic before flux computation
- **Output**: HDF5 + XDMF for Paraview

---

## Non-Goals

- Predefined example shortcuts (use builder)
- C++ implementation initially (Python first, profile later)
- Symbolic equation parsing (keep it simple)
- Parallel time-stepping (future work)

## Version Target

Target: Sampai v0.4.0 or v0.5.0
