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

#### 1.1 Geometry Specification

**Option A: Pre-built Mesh (Flexible)**
```python
import sampai as sam

# User creates mesh separately (full control)
box = sam.geometry.box([-1.0, -1.0], [1.0, 1.0])
mesh = sam.mesh.make(box, min_level=5, max_level=9)

sim = (
    sam.SimulationBuilder(mesh)
    .scheme('rk3', cfl=0.95)
    .solution('u', init='hat')
    .time(tf=1.0, dt=None)
    .build()
)
```

**Option B: Box via Builder (Convenient)**
```python
sim = (
    sam.SimulationBuilder()
    .box([-1.0, -1.0], [1.0, 1.0], min_level=5, max_level=9)
    .scheme('rk3', cfl=0.95)
    .solution('u', init='hat')
    .time(tf=1.0)
    .build()
)
```

**Option C: Complex Domain with Obstacles**
```python
domain = sam.geometry.DomainBuilder2D([-1.0, -1.0], [1.0, 1.0])
domain.remove([0.0, 0.0], [0.4, 0.4])  # Create obstacle

sim = (
    sam.SimulationBuilder()
    .domain(domain, min_level=5, max_level=9)
    .scheme('rk3', rhs=lambda u: sam.operators.convection_weno5(u))
    .solution('u', init='hat')
    .time(tf=1.0)
    .build()
)
```

#### 1.2 Full Builder Example

```python
sim = (
    sam.SimulationBuilder()
    .box([-1.0, -1.0], [1.0, 1.0], min_level=5, max_level=9)
    .scheme('rk3', cfl=0.95)
    .solution('u', init='hat')
    .time(tf=1.0, dt=None)
    .adapt(epsilon=2e-4, frequency='every')
    .output('./results', interval=0.1, format='hdf5')
    .checkpoint('./checkpoints', interval=1.0)
    .progress(desc='Burgers 2D')
    .build()
)
```

#### Builder Methods

| Method | Parameters | Description |
|--------|------------|-------------|
| `.box()` | `min_corner, max_corner, min_level, max_level, ...` | Create simple box domain |
| `.domain()` | `DomainBuilder, min_level, max_level, ...` | Create mesh from complex domain (with obstacles) |
| `__init__(mesh)` | `mesh: MRMesh` | Use pre-built mesh (Option A) |
| `.scheme()` | `name: str` ('euler', 'rk3') or `Scheme` instance, `**kwargs` | Time-stepping scheme with optional params (cfl, rhs) |
| `.solution()` | `name: str`, `init: float|str|callable` | Main solution field, auto-created |
| `.time()` | `tf: float`, `dt: float=None`, `cfl: float` | Time configuration |
| `.adapt()` | `epsilon: float`, `frequency: str|int|callable` | MRA config, 'every', N, or condition(t,iter) |
| `.output()` | `path: str`, `interval: float`, `format: str` | Auto-output configuration |
| `.checkpoint()` | `path: str`, `interval: float=None` | Checkpoint configuration (None = manual only) |
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

@sim.on_checkpoint
def save_additional_data(u, t, iteration, checkpoint_path):
    # Save additional data alongside checkpoint
    np.save(f'{checkpoint_path}/custom_data.npy', {'t': t, 'iter': iteration})
```

| Hook | Signature | When Called |
|------|-----------|-------------|
| `@before_step` | `func(u, t, iteration)` | Before time step |
| `@after_step` | `func(u, t, iteration)` | After time step |
| `@before_adapt` | `func(u)` | Before mesh adaptation |
| `@after_adapt` | `func(u, mesh_stats)` | After mesh adaptation |
| `@on_output` | `func(u, t, iteration)` | When output is saved |
| `@on_checkpoint` | `func(u, t, iteration, checkpoint_path)` | When checkpoint is saved |
| `@on_restart` | `func(checkpoint_path)` | After loading from checkpoint |

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

### 4. Execution Methods

```python
# Full auto-run
u_final = sim.run()

# Step-by-step control
with sim:
    while sim.running:
        u = sim.step()
        # custom logic between steps

# Manual checkpoint during step loop
with sim:
    while sim.running:
        u = sim.step()
        if sim.t > 5.0:
            sim.save_checkpoint()  # Manual checkpoint
        if some_condition:
            break
    sim.save_checkpoint('final')  # Save final state
```

| Method | Returns | Description |
|--------|---------|-------------|
| `.run()` | `Field` | Runs complete simulation, returns final solution |
| `.step()` | `Field` | Executes single time step with AMR |
| `.save_checkpoint()` | `str` | Save current state to checkpoint file, returns path |
| `.load_checkpoint()` | - | Load state from checkpoint file |
| `__enter__()` | `Simulation` | Context manager entry |
| `__exit__()` | - | Context manager exit |

---

### 5. Checkpoint & Restart

#### 5.1 Automatic Checkpoints

```python
sim = (
    sam.SimulationBuilder()
    .box([-1, -1], [1, 1], min_level=5, max_level=9)
    .scheme('rk3')
    .solution('u', init='hat')
    .time(tf=10.0)
    .checkpoint('./checkpoints', interval=1.0)  # Every 1.0 time unit
    .build()
)

u_final = sim.run()
# Checkpoints auto-saved to:
# ./checkpoints/checkpoint_0001.0000.h5
# ./checkpoints/checkpoint_0002.0000.h5
# ...
```

#### 5.2 Manual Checkpoints

```python
with sim:
    while sim.running:
        u = sim.step()

        # Manual checkpoint based on condition
        if u.max() > 0.9:
            path = sim.save_checkpoint('near_shock')  # ./checkpoints/near_shock_t=0.543.h5
```

#### 5.3 Restart from Checkpoint

```python
# Resume from last checkpoint
sim = sam.Simulation.load('./checkpoints/checkpoint_0050.0000.h5')
# Simulation state fully restored: u, t, iteration, mesh, config

# Change final time and continue
sim.config['time']['tf'] = 20.0  # Extend simulation
u_final = sim.run()
```

#### 5.4 Checkpoint Content

Each checkpoint file contains:
- Solution field `u` (HDF5 dataset)
- Current time `t`
- Current iteration count
- Mesh state (cell structure, levels)
- MRA configuration
- Scheme metadata
- User-defined custom data (via `@on_checkpoint`)

---

### 6. Runtime Reconfiguration

#### 6.1 Simple Parameter Changes

```python
sim = sam.SimulationBuilder()... .build()

with sim:
    while sim.running:
        u = sim.step()

        # Dynamically adjust adaptation threshold
        if u.max() > 0.95:  # Near shock
            sim.set_adapt_epsilon(1e-4)  # Refine more
        else:
            sim.set_adapt_epsilon(2e-4)  # Standard

        # Change CFL based on stability
        if u.max() > 1.0:
            sim.set_cfl(0.5)  # Reduce for stability
        else:
            sim.set_cfl(0.95)  # Standard
```

#### 6.2 Conditional Adaptation Control

```python
# Initially adapt every step
sim = sam.SimulationBuilder()... .adapt(epsilon=2e-4, frequency='every').build()

with sim:
    while sim.running:
        u = sim.step()

        # Switch adaptation frequency based on solution smoothness
        if sim.iteration == 100 and u.max() < 0.5:
            # Solution smoothed out, adapt less frequently
            sim.set_adapt_frequency(5)  # Every 5 steps

        # Or use custom condition
        sim.set_adapt_condition(lambda t, it: it % 10 == 0)
```

#### 6.3 Scheme Swapping (Advanced)

```python
# Start with Euler for rapid initial transient
sim = sam.SimulationBuilder()... .scheme('euler', rhs=my_rhs).build()

with sim:
    while sim.running:
        u = sim.step()

        # Switch to RK3 for accuracy after initial transient
        if sim.t > 1.0 and isinstance(sim.stepper, EulerStepper):
            sim.set_scheme('rk3', rhs=my_rhs)
            print(f"Switched to RK3 at t={sim.t}")
```

#### 6.4 Hook-Based Control

```python
@sim.before_step
def adaptive_control(u, t, iteration):
    # Hook can influence simulation behavior
    if u.max() > 1.0:
        # Reduce dt for stability
        sim.dt = 0.5 * sim.dt
    elif u.max() < 0.1:
        # Increase dt for efficiency
        sim.dt = 1.5 * sim.dt
```

---

### 7. Custom Schemes (Extensibility)

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
- **Checkpointing**: Opt-in via `.checkpoint()` builder method
- **Reconfiguration**: Allowed at runtime via setter methods

---

## Advanced Features Summary

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Geometry Options** | Pre-built mesh, Box helper, DomainBuilder with obstacles | Simple to complex domains |
| **Checkpoint/Restart** | Auto or manual checkpointing, full state restoration | Long simulations, crash recovery |
| **Runtime Reconfiguration** | Dynamic parameter changes, scheme swapping, adaptation control | Adaptive algorithms, multi-stage simulations |
| **Hook System** | 7 hooks (@before_step, @after_step, @before_adapt, @after_adapt, @on_output, @on_checkpoint, @on_restart) | Custom logic injection |
| **Progress Tracking** | Auto mesh statistics, progress bar, ETA | User feedback, debugging |

---

## Non-Goals

- Predefined example shortcuts (use builder)
- C++ implementation initially (Python first, profile later)
- Symbolic equation parsing (keep it simple)
- Parallel time-stepping (future work)
- Automatic mesh deformation (mesh topology fixed, only refinement changes)

## Version Target

Target: Sampai v0.4.0 or v0.5.0
