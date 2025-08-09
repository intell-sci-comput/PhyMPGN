# Cylinder Flow Dataset

This dataset contains simulated flow around a cylinder, stored in HDF5 format, including the geometry, fluid properties, and flow dynamics. The dataset is divided into training and testing sets:

- **Training Set**: `train_cf_4x2000x1598x2.h5` contains 4 trajectories (~195MB).
- **Testing Set**: `test_cf_9x2000x1598x2.h5` contains 9 trajectories (~440MB).

## File Structure

Each HDF5 file contains the following attributes and datasets:

### Attributes
- `f.attrs['x_c'], f.attrs['y_c']`: **float**, coordinates of the cylinder center.
- `f.attrs['r']`: **float**, radius of the cylinder.
- `f.attrs['x_l'], f.attrs['x_r'], f.attrs['y_b'], f.attrs['y_t']`: **float**, boundaries of the computational domain.
- `f.attrs['mu']`: **float**, fluid viscosity.
- `f.attrs['rho']`: **float**, fluid density.

### Datasets
- `f['pos']`: **(n, 2)**, positions of observed nodes.
- `f['mesh']`: **(n_tri, 3)**, triangular mesh of observed nodes.
- `g = f['node_type']`: Node type information.
  - `g['inlet']`: **(n_inlet,)**, indices of nodes on the inlet boundary.
  - `g['cylinder']`: **(n_cylinder,)**, indices of nodes on the cylinder boundary.
  - `g['outlet']`: **(n_outlet,)**, indices of nodes on the outlet boundary.
  - `g['inner']`: **(n_inner,)**, indices of nodes within the domain.
- `g = f[i]`: The i-th trajectory.
  - `g['U']`: **(t, n, 2)**, velocity states.
  - `g['dt']`: **float**, time interval between time steps.
  - `g['u_m']`: **float**, inlet velocity.