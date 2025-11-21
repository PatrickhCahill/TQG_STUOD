**STRSW / TQG — PDEs and weak forms**

Short extract of the PDEs implemented in `strsw/solver.py`. Equations use KaTeX/LaTeX and match the variational forms in the code.

**Notation**
- $\psi$: streamfunction (CG)
- $q$: potential vorticity (DG)
- $b$: buoyancy (DG)
- $h$: bathymetry (called `bathymetry` in code)
- $f$: planetary vorticity / rotation (called `rotation` in code)
- $u = \nabla^{\perp}\psi = (-\partial_y\psi,\; \partial_x\psi)$

**Streamfunction (elliptic / diagnostic)**
The code solves the Helmholtz/elliptic problem (variationally written with test function $\phi$):
$$
\int \big(\nabla\phi\cdot\nabla\psi + \phi\,\psi\big)\,dx \,=\, \int \phi\,(f - q)\,dx
$$
which corresponds to the strong form
$$
(1 - \Delta)\,\psi \,=\, f - q.
$$
Reference in code: `Apsi = (dot(grad(phi), grad(psi)) + phi * psi) * dx` and `Lpsi = (self.rotation - self.q1) * phi * dx`.

**Sea-surface-height (SSH) — diagnostic**
The model defines SSH diagnostically (projection into `Vcg`):
$$
\mathrm{ssh} \,=\, \psi - \tfrac{1}{2} b
$$
In the code this appears as e.g. `self.ssh.assign(self.psi0 - 0.5 * Function(self.Vcg).project(b0))`.

**Buoyancy $b$ (advection / transport)**
The discrete weak form used implements conservative DG advection with upwind fluxes. The strong PDE is the transport equation
$$
\partial_t b + \nabla\cdot( u \, b ) \,=\, 0,
$$
or for divergence-free $u$ equivalently
$$
\partial_t b + u\cdot\nabla b \,=\, 0.
$$
Weak form (symbolic):
$$
\int p\, b\,dx \; - \; \Delta t\int \nabla p\cdot (u\, b)\,dx \; - \; \Delta t\,\text{(upwind interface flux terms)} \,=\, \text{RHS mass term},
$$
with the code variables `a_mass_b`, `a_int_b` and `a_flux_b` building these terms.

Reference in code: `a_mass_b = p * b * dx`, `a_int_b = (dot(grad(p), -self.gradperp(self.psi0) * b)) * dx`, and
`a_flux_b = 0.5*jump(p)*(2*_un_('+')*avg(b) + _abs_un_('+')*jump(b))*dS`.

**Potential vorticity $q$ (advection + bathymetry coupling)**
The PV is updated in conservative form with a bathymetry-coupling/source. A compact strong form consistent with the weak form in the code is:
$$
\partial_t q + \nabla\cdot\big( u\,(q - b) \big) + \nabla\cdot\Big( b\,\nabla^{\perp}\big(\tfrac{1}{2}h\big) \Big) \,=\, 0,
$$
where the last term expresses the interaction with bathymetry $h$ (coded as `self.bathymetry`). The code assembles this in weak form as
$$
\int p\, q\,dx \; - \; \Delta t\left( \int \nabla p\cdot u\,(q - b)\,dx \; + \; \int p\,\nabla\cdot\big( b\,\nabla^{\perp}(\tfrac{1}{2}h) \big)\,dx \right) \,=\, \text{mass RHS} ,
$$
plus the DG upwind flux corrections at element interfaces (see `a_flux_`).

Reference in code: `a_mass_ = p_ * q * dx`,
`a_int_ = ( dot(grad(p_), -self.gradperp(self.psi0)*(q - self.db1)) + p_ * div(self.db2 * gradperp(0.5*self.bathymetry)) ) *dx`,
and `a_flux_ = 0.5*jump(p_)*(2*_un_('+')*avg(q-self.db1) + _abs_un_('+')*jump(q - self.db1))*dS`.

**Time-stepping**
The code uses an explicit SSPRK(3) style update (three-stage low-storage RK). At each stage the elliptic problem for $\psi$ is solved diagnostically using the current $q$; then $b$ and $q$ are advanced via DG mass + advective terms with upwind fluxes.

**Implementation notes / mapping to code**
- `psi` & Helmholtz solve: `Apsi` / `Lpsi` near top of `STQGSolver.__init__`.
- `b` DG advection: `a_mass_b`, `a_int_b`, `a_flux_b`, `b_solver`.
- `q` DG advection + bathymetry: `a_mass_`, `a_int_`, `a_flux_`, `q_solver`.
- `ssh` is diagnostic: `self.ssh.assign(self.psi0 - 0.5 * Function(self.Vcg).project(b0))`.

If you want this file moved, extended with step-by-step integration-by-parts derivations, or exported as PDF/LaTeX, tell me which format you prefer.
