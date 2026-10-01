# 6h I0B0 N two-RHS record audit

The protocol was frozen and committed before this one paired attempt. The
original 6h protocol, historical production source, input arrays and archived
checkpoints are unchanged. New source observations are isolated in a disposable
private copy. The patch and 35-module hash manifest are portable records; private
arrays are excluded from publication.

The exact sequence is tracer RK update, first residual at the incoming N state,
implicit linear-bottom-drag compensation, Euler predictor with updated tracer
T/S, second residual at that predictor, second drag compensation, and the
original Heun velocity update. Both actual evaluation states, full tendency,
ordered residual intermediates, raw residual and compensated RHS are recorded.
Removing auxiliary AST nodes exactly restores the historical module AST.

Every component increment is dt/2 times its two actual RHS terms. Every work
uses the same actual incoming/outgoing N velocity midpoint and original fixed
reference mass. The internal Euler state is not an independently accepted
energy jump. Raw and filtered advection use the same wet mask before taking the
filter difference. The final residual mask is its observed map; remaining
arithmetic error is reported without assigning it to a physical source.

The fixed protocol requires paired finite six-field attempted/returned error
at most 1e-11, identical original booleans, original acceptance, reconstruction
of each actual residual transition, compensated RHS, final increment and
fixed-weight work. A failed gate prevents attribution. No positivity or damping
assumption is imposed on tendency dealiasing.

Reference kinetic work and the change of moving-top-minus-reference kinetic
energy are distinct diagnostics. Compatible density/buoyancy PE remains
unproven. This record does not claim full energy closure, scientific quality,
initial-state repair, or a 30h failure remedy.
